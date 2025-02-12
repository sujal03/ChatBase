import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# LangChain and ChromaDB imports
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

load_dotenv()

# ---------------------------
# Configuration
# ---------------------------
PDF_DIR = "uploads/"
CHROMA_PERSIST_DIR = "./chroma_db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# PDF Indexing Functions
# ---------------------------
def load_pdfs_from_directory(directory):
    """
    Load all PDFs from the given directory and add source metadata.
    """
    pdf_documents = []
    directory = Path(directory)
    for pdf_path in directory.glob('*.pdf'):
        try:
            loader = PDFPlumberLoader(str(pdf_path))
            documents = loader.load()
            for doc in documents:
                doc.metadata = doc.metadata or {}
                doc.metadata["source"] = pdf_path.name
            pdf_documents.extend(documents)
            logging.info(f"✅ Loaded: {pdf_path.name}")
        except Exception as e:
            logging.exception(f"Error loading {pdf_path.name}: {e}")
    return pdf_documents

def process_documents(documents):
    """
    Split documents into smaller chunks for indexing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    try:
        chunks = text_splitter.split_documents(documents)
        logging.info(f"✅ Processed documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logging.exception("Error during document splitting")
        raise

def index_to_chroma(chunks):
    """
    Index document chunks into ChromaDB.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(
        collection_name="documents",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    vectordb.add_documents(chunks)
    logging.info("✅ Successfully indexed documents!")
    return vectordb

def create_and_index_embeddings():
    """
    Load PDFs from the PDF_DIR, split them into chunks, and index them into ChromaDB.
    After indexing, delete the processed PDF files.
    """
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    logging.info("🔍 Loading PDFs...")
    raw_docs = load_pdfs_from_directory(PDF_DIR)
    if not raw_docs:
        logging.warning("⚠️ No PDF documents found!")
        return None
    
    logging.info(f"📄 Found {len(raw_docs)} document(s). Processing...")
    chunks = process_documents(raw_docs)
    
    logging.info("📥 Indexing document chunks to ChromaDB...")
    vectordb_local = index_to_chroma(chunks)
    logging.info(f"✅ Successfully indexed {len(chunks)} chunk(s) to ChromaDB")
    
    # Delete PDFs after indexing
    for pdf_path in Path(PDF_DIR).glob('*.pdf'):
        try:
            pdf_path.unlink()
            logging.info(f"🗑️ Deleted file: {pdf_path.name}")
        except Exception as e:
            logging.error(f"❌ Failed to delete file {pdf_path.name}: {e}")
    
    return vectordb_local

# ---------------------------
# Components for RAG & Chatbot
# ---------------------------
def init_components():
    """
    Initialize vector database and Gemini LLM using the provided GOOGLE_API_KEY.
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing. Please check your .env file.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectordb = Chroma(
        collection_name="documents",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        streaming=True,
        temperature=0.5
    )
    return vectordb, embeddings, llm

vectordb, embeddings, llm = init_components()

# ---------------------------
# RAG Prompt Templates
# ---------------------------
PROMPT_TEMPLATE = """
You are an AI assistant helping users extract information from documents.
Use only the provided context to answer questions.

Context:
{context}

User Question:
{question}

### Answer Guidelines:
- Be clear and concise.
- Use points for structured answers.
- Avoid assumptions or fabrications.
- Provide accurate and relevant information.
- If the answer is not in the context, you can say "The answer is not in the document."
- don't mention the document or the context in the answer.
"""

FALLBACK_PROMPT = """
You are an AI assistant specializing in general purpose assistance.
Since there is no relevant document-based context, answer the question using your knowledge base.

User Question:
{question}

### Answer Guidelines:
- Keep the response **brief** and **to the point**.
- Provide accurate and relevant information.
- Avoid assumptions, opinions, or unrelated information.
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
fallback_prompt = ChatPromptTemplate.from_template(FALLBACK_PROMPT)

def search_documents(user_query, vectordb, embeddings, k=5, threshold=0.8):
    results = vectordb.similarity_search_with_score(user_query, k=k)
    relevant_docs = [doc[0] for doc in results if doc[1] < threshold]
    return relevant_docs

def generate_llm_response(user_query, vectordb, llm, threshold=0.8):
    relevant_docs = search_documents(user_query, vectordb, embeddings, threshold=threshold)

    context = "\n\n".join(doc.page_content for doc in relevant_docs) 
    final_prompt = prompt  # Optionally use fallback_prompt if context is empty
    prompt_data = {"context": context, "question": user_query}

    chain = final_prompt | llm
    response_chunks = []
    for chunk in chain.stream(prompt_data):
        response_chunks.append(chunk.content)
    return "".join(response_chunks)

# ---------------------------
# New Function: Add Q&A Input to Vector Database for RAG
# ---------------------------
def add_qa_to_vector_database(question: str, answer: str):
    qa_content = f"Question: {question}\nAnswer: {answer}"
    doc = Document(page_content=qa_content, metadata={"source": "user_qa"})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = text_splitter.split_documents([doc])
    vectordb.add_documents(docs)
    logging.info("✅ Q&A pair successfully indexed into the vector database.")

# ---------------------------
# Existing Function: Add Text Input to Vector Database for RAG
# ---------------------------
def add_text_to_vector_database(text_data: str):
    doc = Document(page_content=text_data, metadata={"source": "user_text_input"})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = text_splitter.split_documents([doc])
    vectordb.add_documents(docs)
    logging.info("✅ User text successfully indexed into the vector database.")

# ---------------------------
# Optional: Testing when run as a script
# ---------------------------
if __name__ == "__main__":
    # Test PDF indexing and deletion
    vectordb_from_indexing = create_and_index_embeddings()
    if vectordb_from_indexing:
        test_query = "Your detailed search query here"
        results = search_documents(test_query, vectordb_from_indexing, embeddings)
        for idx, doc in enumerate(results, start=1):
            logging.info(f"Result {idx}: {doc.page_content[:100]}...")

    # Test LLM response
    response = generate_llm_response("Tell me about the document.", vectordb, llm)
    logging.info(f"LLM Response: {response}")

    # Test indexing text input for RAG
    sample_text = "This is a sample text input for testing retrieval augmented generation."
    add_text_to_vector_database(sample_text)

    # Test indexing Q&A input for RAG
    sample_question = "What is the capital of France?"
    sample_answer = "The capital of France is Paris."
    add_qa_to_vector_database(sample_question, sample_answer)
