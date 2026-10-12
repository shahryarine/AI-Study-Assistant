import logging
import requests
import streamlit as st

class StreamlitUIHandler(logging.Handler):
    def __init__(self, ui_container):
        super().__init__()
        self.ui_container = ui_container

    def emit(self, record):
        msg = self.format(record)
        self.ui_container.markdown(f"🔹 {msg}")

def handle_ui_error(error: Exception):
    error_msg = str(error).lower()
    if isinstance(error, requests.exceptions.ConnectionError) or "connection" in error_msg:
        st.error("🔌 Network Error: Please check your internet connection and try again.")
    elif "429" in error_msg or "rate limit" in error_msg:
        st.warning("⏳ The AI server is currently busy. Please wait a few seconds and retry.")
    elif "timeout" in error_msg:
        st.warning("⏱️ The request took too long. The server might be overloaded.")
    elif isinstance(error, ValueError):
        st.warning(f"⚠️ Invalid Input: {str(error)}")
    else:
        st.error("❌ An unexpected error occurred. Please try refreshing the page.")
        logging.error(f"Unhandled UI Exception: {str(error)}")

def attach_ui_logger(status_container) -> StreamlitUIHandler:
    ui_handler = StreamlitUIHandler(status_container)
    ui_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(ui_handler)
    logging.getLogger().setLevel(logging.INFO)
    return ui_handler

def detach_ui_logger(ui_handler: StreamlitUIHandler):
    logging.getLogger().removeHandler(ui_handler)