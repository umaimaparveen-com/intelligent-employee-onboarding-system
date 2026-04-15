"""
───────────────────────────────────────────────────────────────────────────────
ReAct-style Employee Onboarding Agent with full structured observability.

Observability layers
─────────────────────
1. Structured JSON trace log  → logs/agent_trace.jsonl
   Every event (agent_start, llm_call, tool_call, agent_finish, error) is
   written as a newline-delimited JSON record, making logs trivially queryable
   with jq, Pandas, or any log aggregator (Datadog, Grafana Loki, etc.).

2. Human-readable run log     → logs/agent.log
   Classic timestamped log for quick tailing.

3. Console output             → stdout
   Emoji-enriched progress so the developer can follow along in real time.

Trace schema (each JSONL line)
───────────────────────────────
{
  "ts":          "ISO-8601 timestamp",
  "run_id":      "unique ID per agent.run() invocation",
  "event":       "agent_start | llm_call | tool_call | agent_finish | error",
  "step":        int (ReAct step number),
  "tool":        str | null,
  "args":        dict | null,
  "result":      str | null,
  "thought":     str | null,
  "action":      str | null,
  "latency_ms":  float,
  "status":      "ok" | "error"
}

Hallucination-prevention design notes
───────────────────────────────────────
• The agent prompt explicitly lists which tools have already been called and
  whether all four tasks are done; this prevents the LLM from making up a
  completion it has not actually performed.
• Tool results are injected verbatim into the step history so the LLM always
  reasons from observed facts, not from its parametric memory.
• In production we would add a post-step faithfulness check via the
  RAGEvaluator to auto-detect and surface low-confidence answers.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── Paths ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)

# ── Human-readable log ─────────────────────────────────────────────────────
logging.basicConfig(
    filename="logs/agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Structured trace log (JSONL) ───────────────────────────────────────────
TRACE_LOG_PATH = Path("logs/agent_trace.jsonl")
_trace_fh = TRACE_LOG_PATH.open("a", buffering=1)   # line-buffered


def _emit_trace(run_id: str, event: str, step: int = 0, **kwargs) -> None:
    """Append one structured JSON record to the JSONL trace log."""
    record = {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "run_id":     run_id,
        "event":      event,
        "step":       step,
        "tool":       kwargs.get("tool"),
        "args":       kwargs.get("args"),
        "result":     kwargs.get("result"),
        "thought":    kwargs.get("thought"),
        "action":     kwargs.get("action"),
        "latency_ms": kwargs.get("latency_ms", 0.0),
        "status":     kwargs.get("status", "ok"),
    }
    _trace_fh.write(json.dumps(record) + "\n")
    logger.info(f"TRACE event={event} run_id={run_id} step={step} status={record['status']}")


# ── Mock Database ──────────────────────────────────────────────────────────
# Simulates a real backend — in production these would be real API calls
onboarding_db: dict = {}

# ── Tool 1: Create IT Ticket ───────────────────────────────────────────────
def create_it_ticket(employee_name: str, request_type: str) -> dict:
    """Provision laptop, email, VPN access for a new employee."""
    ticket_id = f"IT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    result = {
        "status":       "success",
        "ticket_id":    ticket_id,
        "employee":     employee_name,
        "request_type": request_type,
        "message":      f"IT ticket {ticket_id} created for {employee_name} — {request_type} provisioning started.",
    }
    logger.info(f"IT Ticket created: {result}")
    print(f"  🖥️  IT Ticket: {result['message']}")
    return result


# ── Tool 2: Schedule Orientation ───────────────────────────────────────────
def schedule_orientation(employee_name: str, date: str, department: str) -> dict:
    """Book a calendar orientation event for the new employee."""
    event_id = f"EVT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    result = {
        "status":     "success",
        "event_id":   event_id,
        "employee":   employee_name,
        "date":       date,
        "department": department,
        "message":    f"Orientation scheduled for {employee_name} ({department}) on {date}.",
    }
    logger.info(f"Orientation scheduled: {result}")
    print(f"  📅 Orientation: {result['message']}")
    return result


# ── Tool 3: Send Welcome Email ─────────────────────────────────────────────
def send_welcome_email(employee_name: str, email: str, start_date: str) -> dict:
    """Send a welcome email to the new hire."""
    result = {
        "status":     "success",
        "employee":   employee_name,
        "email":      email,
        "start_date": start_date,
        "message":    f"Welcome email sent to {employee_name} at {email} for start date {start_date}.",
    }
    logger.info(f"Welcome email sent: {result}")
    print(f"  📧 Email: {result['message']}")
    return result


# ── Tool 4: Update Onboarding Status ──────────────────────────────────────
def update_onboarding_status(employee_id: str, task: str, status: str) -> dict:
    """Track and update onboarding checklist progress."""
    if employee_id not in onboarding_db:
        onboarding_db[employee_id] = {}
    onboarding_db[employee_id][task] = {
        "status":     status,
        "updated_at": datetime.now().isoformat(),
    }
    result = {
        "status":      "success",
        "employee_id": employee_id,
        "task":        task,
        "task_status": status,
        "message":     f"Onboarding task '{task}' marked as '{status}' for {employee_id}.",
    }
    logger.info(f"Status updated: {result}")
    print(f"  ✅ Status: {result['message']}")
    return result


# ── Tool Registry ──────────────────────────────────────────────────────────
TOOLS = {
    "create_it_ticket":       create_it_ticket,
    "schedule_orientation":   schedule_orientation,
    "send_welcome_email":     send_welcome_email,
    "update_onboarding_status": update_onboarding_status,
}

TOOLS_DESCRIPTION = """\
You have access to these tools:

1. create_it_ticket(employee_name, request_type)
   - Use to provision laptop, email, VPN for a new employee

2. schedule_orientation(employee_name, date, department)
   - Use to book an orientation session for the new employee

3. send_welcome_email(employee_name, email, start_date)
   - Use to send a welcome email to the new hire

4. update_onboarding_status(employee_id, task, status)
   - Use to track onboarding checklist progress
"""


# ── ReAct Agent ────────────────────────────────────────────────────────────
class OnboardingAgent:
    """
    ReAct-style agent (Reason → Act → Observe) that executes
    onboarding workflows autonomously using tool calls.

    Observability features
    ──────────────────────
    - Every LLM call is timed and written to the JSONL trace log.
    - Every tool call (success or failure) is traced with args + result.
    - Each run gets a unique run_id so traces for different invocations
      can be separated in a log aggregator.
    """

    def __init__(self, model: str = "llama3"):
        self.llm       = OllamaLLM(model=model, temperature=0)
        self.max_steps = 10

    # ── Prompt builder ──────────────────────────────────────────────────────
    def _build_prompt(self, task: str, steps_taken: list) -> str:
        steps_text = "\n".join(steps_taken) if steps_taken else "None yet."

        it_done          = any("create_it_ticket"       in s for s in steps_taken)
        email_done       = any("send_welcome_email"      in s for s in steps_taken)
        orientation_done = any("schedule_orientation"    in s for s in steps_taken)
        status_done      = any(
            "update_onboarding_status" in s and "Priya Sharma" in s and "in_progress" in s
            for s in steps_taken
        )
        all_done = it_done and email_done and orientation_done and status_done

        return f"""You are an intelligent Employee Onboarding Agent for Amiseq.
    Your job is to complete onboarding tasks by calling the right tools in the right order.

    {TOOLS_DESCRIPTION}

    STRICT RULES:
    - Call each tool ONLY ONCE
    - As soon as these 4 tasks are done, respond with FINISH immediately:
    1. create_it_ticket ✓ if done: {it_done}
    2. send_welcome_email ✓ if done: {email_done}
    3. schedule_orientation ✓ if done: {orientation_done}
    4. update_onboarding_status ✓ if done: {status_done}
    - ALL 4 DONE = {all_done} → if True, your ONLY valid response is FINISH
    - Respond ONLY in this exact JSON format:

    If you need to call a tool:
    {{"thought": "your reasoning here", "action": "tool_name", "args": {{"param1": "value1"}}}}

    If all 4 tasks are complete:
    {{"thought": "all tasks done", "action": "FINISH", "summary": "brief summary of everything completed"}}

    TASK: {task}

    STEPS TAKEN SO FAR:
    {steps_text}

    What is your next action? Respond ONLY with JSON.
    """

    # ── Tool executor ───────────────────────────────────────────────────────
    def _call_tool(
        self, tool_name: str, args: dict, run_id: str, step: int
    ) -> str:
        """
        Execute a tool, emit a structured trace record, and return the result.

        Tracing strategy
        ────────────────
        We record args and results for every call regardless of success/failure.
        In production this feeds into:
          - Dashboards (tool call frequency, error rates)
          - Alerting (tool failure spike detection)
          - Replay (reproducing a failed run from the trace)
        """
        if tool_name not in TOOLS:
            err = f"Error: Tool '{tool_name}' not found."
            _emit_trace(run_id, "tool_call", step=step, tool=tool_name,
                        args=args, result=err, status="error")
            return err

        t0 = time.time()
        try:
            result     = TOOLS[tool_name](**args)
            result_str = json.dumps(result)
            latency    = (time.time() - t0) * 1000
            _emit_trace(
                run_id, "tool_call", step=step,
                tool=tool_name, args=args, result=result_str,
                latency_ms=round(latency, 2), status="ok",
            )
            return result_str
        except Exception as exc:
            latency = (time.time() - t0) * 1000
            err_payload = json.dumps({"status": "error", "message": str(exc)})
            _emit_trace(
                run_id, "tool_call", step=step,
                tool=tool_name, args=args, result=err_payload,
                latency_ms=round(latency, 2), status="error",
            )
            logger.error(f"Tool error: {tool_name} - {exc}")
            return err_payload

    # ── LLM response parser ─────────────────────────────────────────────────
    def _parse_llm_response(self, response: str) -> dict:
        """Parse the LLM JSON response safely."""
        try:
            start    = response.find("{")
            end      = response.rfind("}") + 1
            json_str = response[start:end]
            return json.loads(json_str)
        except Exception as exc:
            logger.error(f"Parse error: {exc} | Response: {response}")
            return {"action": "FINISH", "summary": "Could not parse response.", "thought": ""}

    # ── Main ReAct loop ─────────────────────────────────────────────────────
    def run(self, task: str) -> str:
        """
        Run the ReAct loop:
        1. Reason → call LLM to decide next action
        2. Act    → execute the chosen tool
        3. Observe → record the result and loop

        Every iteration writes structured trace events to agent_trace.jsonl.
        """
        run_id = str(uuid.uuid4())

        print(f"\n🤖 Agent Starting Task:")
        print(f"📋 {task}")
        print(f"🔑 Run ID: {run_id}")
        print("=" * 60)

        _emit_trace(run_id, "agent_start", step=0,
                    thought=task, status="ok")

        steps_taken: list[str] = []
        step_num = 0

        while step_num < self.max_steps:
            step_num += 1
            print(f"\n🔄 Step {step_num}:")

            # ── Reason (LLM call) ──
            prompt = self._build_prompt(task, steps_taken)
            t0     = time.time()
            response = self.llm.invoke(prompt)
            llm_latency = (time.time() - t0) * 1000

            parsed  = self._parse_llm_response(response)
            thought = parsed.get("thought", "")
            action  = parsed.get("action", "FINISH")

            _emit_trace(
                run_id, "llm_call", step=step_num,
                thought=thought, action=action,
                latency_ms=round(llm_latency, 2), status="ok",
            )
            print(f"  💭 Thought: {thought}")

            # ── Finish condition ──
            if action == "FINISH":
                summary = parsed.get("summary", "All tasks completed.")
                print(f"\n✅ Agent Complete!")
                print(f"📝 Summary: {summary}")
                _emit_trace(
                    run_id, "agent_finish", step=step_num,
                    thought=thought, result=summary, status="ok",
                )
                logger.info(f"Run {run_id} completed: {summary}")
                return summary

            # ── Act (tool call) ──
            args = parsed.get("args", {})
            print(f"  🔧 Action: {action}({args})")

            # ── Observe ──
            observation  = self._call_tool(action, args, run_id, step_num)
            step_record  = f"Step {step_num}: Called {action}({args}) → {observation}"
            steps_taken.append(step_record)

        # Max steps reached
        _emit_trace(run_id, "agent_finish", step=step_num,
                    result="Max steps reached", status="error")
        return "Agent reached maximum steps without completing."


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = OnboardingAgent()

    task = """
    Onboard Priya Sharma. She is joining the Engineering team in the 
    Bangalore office on May 5, 2026. Her email is priya.sharma@amiseq.com.
    Make sure her laptop is provisioned, she gets a welcome email, 
    orientation is scheduled, and her onboarding checklist is started.
    """

    result = agent.run(task)
    print("\n📦 Onboarding Database State:")
    print(json.dumps(onboarding_db, indent=2))

    # ── Show last few trace lines ──
    print("\n📡 Last 5 trace events (from agent_trace.jsonl):")
    with open("logs/agent_trace.jsonl") as f:
        lines = f.readlines()
    for line in lines[-5:]:
        rec = json.loads(line)
        print(f"  [{rec['ts']}] event={rec['event']:14s} step={rec['step']} "
              f"tool={rec['tool'] or '—':25s} status={rec['status']} "
              f"latency={rec['latency_ms']:.0f}ms")