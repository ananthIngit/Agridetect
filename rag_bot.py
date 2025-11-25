# ==============================================================================
# 1. IMPORT NECESSARY LIBRARIES (LCEL CHAIN STRUCTURE)
# ==============================================================================
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW LCEL Imports to build the chain directly:
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# 2. CONFIGURATION AND INITIAL SETUP
# ==============================================================================

# Define the paths and model names
DATA_FILE = "disease_data.txt"
CHROMA_DB_DIR = "./chroma_db"
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"

# --- RAG Orchestration ---
def setup_rag_pipeline():
    """Initializes the Ollama models and sets up the Chroma Vector Store."""
    
    print("--- 1. Initializing Models and Connection ---")
    
    # 1. Ollama LLM Connection (Mistral)
    try:
        llm = Ollama(model=LLM_MODEL)
        
        # 2. Ollama Embedding Model Connection
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
    except Exception as e:
        print("\nFATAL ERROR: Could not connect to Ollama.")
        print(f"Details: {e}")
        print("ACTION REQUIRED: Ensure the Ollama application is running in the background.")
        return None, None
        
    # --- Data Processing ---
    print("--- 2. Processing Knowledge Base ---")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Knowledge file '{DATA_FILE}' not found.")
        print("ACTION REQUIRED: Please create the file and add your disease data.")
        return None, None

    # Split the document into small chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200,
        separators=["\n---\n", "\n\n", "\n", " ", ""] # Use the '---' as a top-level separator
    )
    documents = text_splitter.create_documents([data])
    print(f"-> Data processed and split into {len(documents)} chunks.")

    # --- Vector Store (Chroma) Setup ---
    print("--- 3. Creating/Loading Vector Store ---")
    vectorstore = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    # The retriever object is key for the conditional logic in start.py
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    print(f"-> Vector Store created/loaded successfully in '{CHROMA_DB_DIR}'")

    return llm, retriever

# --- RAG Prompt Template (Modified for Conversational/General Chat) ---
def create_rag_prompt():
    """Defines the specialized prompt template for the AgriBot."""
    
    template = """
    You are AgriBot, a friendly and experienced AI assistant specializing in agriculture and general conversation.
    Your tone should be engaging, encouraging, and supportive (like a friend).

    GUIDANCE:
    1. **If the QUESTION is related to plant diseases or farming practices covered by the CONTEXT,** you MUST use the CONTEXT to formulate your clear, actionable response.
    2. **If the QUESTION is general (e.g., "how are you?", "tell me a joke," or general farming questions not in the CONTEXT),** use your vast general knowledge base to answer conversationally.
    3. **Do not repeat disease information unless asked.**

    CONTEXT:
    {context}
    
    QUESTION: {question}
    """
    return PromptTemplate.from_template(template)

# --- Query Execution Function (Used for simple tests/debugging) ---
def run_query(chain, query):
    """Runs the final query through the LCEL chain."""
    print(f"\nExecuting Query: '{query}'...")
    response = chain.invoke(query) 
    return response

# Note: The __main__ block is removed to avoid running tests when imported by start.py