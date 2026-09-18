
import os
import logging
import streamlit as st

# Configure standard logging for the application
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MUST be the very first Streamlit command in the script
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered"
)

# -------------------------------------------------------------------------
# 1. Resource Management (Memory Leak Prevention)
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI Models... Please wait.")
def load_embedding_model():
    """
    Loads the SentenceTransformer model once and caches it in memory.
    This prevents the model from reloading on every user interaction.
    """
    try:
        # TODO: Import and initialize the Embedder class from Phase 2
        # from rag.embedder import Embedder
        # return Embedder()
        
        logger.info("Embedding model loaded into cache successfully.")
        return "Dummy_Embedder_Ready" 
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        st.error("Critical Error: Could not load the AI model. Check system logs.")
        st.stop()

@st.cache_resource(show_spinner="Connecting to Vector Database...")
def load_vector_store():
    """
    Initializes the ChromaDB persistent client once.
    Ensures a singleton connection pattern across the app lifecycle.
    """
    try:
        # TODO: Import and initialize the VectorStore class from Phase 2
        # from rag.vector_store import VectorStore
        # return VectorStore()
        
        logger.info("Vector database connected successfully.")
        return "Dummy_VectorStore_Ready"
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        st.error("Critical Error: Database connection failed. Check system logs.")
        st.stop()

# -------------------------------------------------------------------------
# 2. Session State Initialization
# -------------------------------------------------------------------------
def init_session_state():
    """
    Initializes global variables to preserve state across page reloads.
    Prevents UI resets when buttons are clicked.
    """
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = False
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "exam_data" not in st.session_state:
        st.session_state.exam_data = []

# -------------------------------------------------------------------------
# 3. Main UI Layout & Logic
# -------------------------------------------------------------------------
def main():
    # Initialize state variables before rendering any UI components
    init_session_state()
    
    # Safely load heavy resources using cached functions
    embedder = load_embedding_model()
    vector_store = load_vector_store()
    
    st.title("📚 AI Study Assistant")
    st.markdown("Upload your **PDF** document to start analyzing, chatting, and generating exams.")
    
    # File uploader restricted specifically to PDF formats
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if not st.session_state.is_processed:
            with st.spinner("Processing document and generating vectors..."):
                try:
                    # Save the uploaded file to a temporary location safely
                    os.makedirs("data/temp", exist_ok=True)
                    temp_path = os.path.join("data/temp", uploaded_file.name)
                    
                    with open(temp_path, "wb") as file_buffer:
                        file_buffer.write(uploaded_file.getbuffer())
                        
                    # TODO: Integrate the Ingestion Pipeline from Phase 2 here
                    # pipeline = Indexer(embedder=embedder, vector_store=vector_store)
                    # pipeline.index_text_file(temp_path)
                    
                    # Lock the processing state to prevent re-execution
                    st.session_state.is_processed = True
                    st.success("Document processed and indexed successfully!")
                    
                except Exception as e:
                    logger.error(f"File processing pipeline failed: {e}")
                    st.error("An error occurred while processing the file. Please try a different PDF.")
        else:
            st.success("Document is currently active in memory.")

if __name__ == "__main__":
    main()