import os
import sys
import logging
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui_utils import attach_ui_logger, detach_ui_logger, handle_ui_error

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="centered")
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Initializing AI Infrastructure...")
def load_core_components():
    from rag.embedder import Embedder
    from rag.vector_store import VectorStore
    from rag.retriever import Retriever
    from rag.generator import OpenRouterClient
    from rag.engine import AnswerEngine
    from rag.pipeline import Pipeline
    from rag.chunker import TextChunker

    try:
        embedder = Embedder()
        vector_store = VectorStore()
        retriever = Retriever(embedder=embedder, vector_store=vector_store)
        llm_client = OpenRouterClient()
        
        pipeline = Pipeline(chunker=TextChunker(), embedder=embedder, vector_store=vector_store)
        engine = AnswerEngine(embedder=embedder, retriever=retriever, llm_client=llm_client)
        
        return pipeline, engine, vector_store
    except Exception as error:
        st.error("Critical System Failure: Unable to load AI backend.")
        logger.critical(f"Initialization Error: {error}")
        st.stop()

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_file" not in st.session_state:
        st.session_state.current_file = None

def main():
    init_session_state()
    pipeline, engine, vector_store = load_core_components()
    
    st.title("📚 AI Study Assistant")
    st.markdown("Upload your **Markdown (.md)** or **Text (.txt)** notes to chat with your knowledge base.")
    
    uploaded_file = st.file_uploader("Upload Notes", type=["md", "txt"])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        if st.session_state.current_file != file_name:
            if vector_store.document_exists(file_name):
                st.success(f"'{file_name}' is already indexed in the database!")
                st.session_state.current_file = file_name
            else:
                with st.status("⚙️ Processing Document...", expanded=True) as status:
                    ui_handler = attach_ui_logger(status)
                    try:
                        os.makedirs("data/temp", exist_ok=True)
                        temp_path = os.path.join("data/temp", file_name)
                        
                        with open(temp_path, "wb") as file_buffer:
                            file_buffer.write(uploaded_file.getbuffer())
                            
                        chunks_indexed = pipeline.process_file(temp_path)
                        
                        st.session_state.current_file = file_name
                        status.update(label=f"✅ Indexed {chunks_indexed} chunks successfully!", state="complete", expanded=False)
                        st.toast("Document is ready for chat!", icon="🎉")
                    except Exception as error:
                        status.update(label="❌ Processing failed", state="error")
                        handle_ui_error(error)
                    finally:
                        detach_ui_logger(ui_handler)

    st.divider()
    st.subheader("Chat with your Notes")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if not prompt.strip():
            st.toast("Please enter a valid question.", icon="⚠️")
            return

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("🔍 Analyzing knowledge base...", expanded=True) as qa_status:
                ui_handler = attach_ui_logger(qa_status)
                answer = ""
                try:
                    answer = engine.ask_question(prompt)
                    qa_status.update(label="✅ Answer generated", state="complete", expanded=False)
                except Exception as error:
                    qa_status.update(label="❌ Generation failed", state="error")
                    handle_ui_error(error)
                    answer = "I encountered an error and could not generate a response."
                finally:
                    detach_ui_logger(ui_handler)
            
            if answer and "encountered an error" not in answer:
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:
        st.error("A fatal application error occurred. Please restart.")
        logger.critical(f"App Crash: {fatal_error}")