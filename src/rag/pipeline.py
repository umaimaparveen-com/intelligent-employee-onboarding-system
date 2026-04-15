import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────
DOCS_PATH = Path("data/docs")
CHROMA_PATH = Path("data/chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"

# ── 1. Load Documents ──────────────────────────────────────────────────────
def load_documents():
    """Load all markdown documents from the docs folder."""
    print("📄 Loading documents...")
    loader = DirectoryLoader(
        str(DOCS_PATH),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents")
    return docs

# ── 2. Chunk Documents ─────────────────────────────────────────────────────
def chunk_documents(docs):
    """
    Split documents into chunks.
    Using RecursiveCharacterTextSplitter because:
    - Respects natural boundaries (paragraphs, sentences)
    - Better semantic coherence than fixed-size chunking
    - Overlap ensures context is not lost between chunks
    """
    print("✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ── 3. Create Vector Store ─────────────────────────────────────────────────
def create_vectorstore(chunks):
    """
    Embed chunks and store in ChromaDB.
    Using sentence-transformers (all-MiniLM-L6-v2) because:
    - Free, no API key needed
    - Fast and lightweight
    - Good performance for semantic similarity
    """
    print("🔢 Creating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH)
    )
    print("✅ Vector store created")
    return vectorstore

# ── 4. Load Existing Vector Store ─────────────────────────────────────────
def load_vectorstore():
    """Load an existing ChromaDB vector store from disk."""
    print("📦 Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )
    return vectorstore

# ── 5. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build the RAG chain with:
    - Retriever: top 3 most relevant chunks
    - Prompt: instructs LLM to use only retrieved context
    - LLM: Llama3 via Ollama (free, local)
    - Fallback: if answer not in docs, say so honestly
    """
    print("🔗 Building RAG chain...")

    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    prompt_template = """
You are a helpful Employee Onboarding Assistant for Amiseq.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I don't have information about that in the onboarding documents. Please contact HR at hr@amiseq.com"
Do NOT make up any information.
Always cite which document or section your answer comes from.

Context:
{context}

Question: {question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain ready")
    return rag_chain

# ── 6. Query ───────────────────────────────────────────────────────────────
def query(rag_chain, question: str):
    """Run a question through the RAG pipeline and print the answer."""
    print(f"\n❓ Question: {question}")
    print("-" * 60)
    result = rag_chain.invoke(question)
    print(f"💬 Answer: {result}")
    return result

# ── 7. Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build the pipeline (only need to run once)
    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorstore = create_vectorstore(chunks)
    rag_chain = build_rag_chain(vectorstore)

    # Test queries from the assignment
    test_questions = [
        "What is the company's remote work policy?",
        "How do I set up VPN on my laptop?",
        "When is the next payroll cycle?",
        "What's the dress code for the Bangalore office?",
        "Who is the CEO of Apple?",  # Out of scope — should decline
    ]

    for question in test_questions:
        query(rag_chain, question)
        print("\n" + "=" * 60 + "\n")