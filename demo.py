import gradio as gr
import requests
import uuid
from typing import Optional
from fastapi import UploadFile
from pydantic_models import DocumentInput, ResetChatHistoryInput
from loguru import logger

API_BASE_URL = "http://localhost:8000"


class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"history": [], "documents": [], "urls": []}
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)

    def reset_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]["history"] = []
            return True
        return False


session_manager = SessionManager()


def process_document(file_obj: UploadFile, url, session_id):
    if session_id is None:
        session_id = session_manager.create_session()

    response_msg = ""
    if file_obj is not None:
        file_name = file_obj.name
        document = DocumentInput(
            docs_url=file_name,
            session_id=session_id,
            config_path="custom_config",
        )
        response = requests.post(
            f"{API_BASE_URL}/upload_pdf",
            json=document.model_dump(),
        )

        if response.status_code == 200:
            response_msg += f"Document uploaded successfully to session {session_id}\n"
        else:
            response_msg += (
                f"Failed to upload document: {response.text}. "
                f"Status code: {response.status_code}\n"
            )

    if url and url.strip():
        document = DocumentInput(
            docs_url=url,
            session_id=session_id,
            config_path="custom_config",
        )
        response = requests.post(
            f"{API_BASE_URL}/upload_url",
            json=document.model_dump(),
        )
        if response.status_code == 200:
            response_msg += f"URL processed successfully in session {session_id}\n"
        else:
            response_msg += f"Failed to process URL: {response.text}\n"

    return response_msg, session_id


def chat(message, history, session_id):
    logger.debug("New message in chat: {}", message)
    logger.debug("Chat history: {}", history)
    logger.debug("Session ID: {}", session_id)

    if session_id is None:
        error_msg = "No active session. Please upload a document first."
        history.append((message, error_msg))
        return "", history

    response = requests.post(
        f"{API_BASE_URL}/chat", json={"question": message, "session_id": session_id}
    )

    if response.status_code == 200:
        logger.debug("Successful response: {}", response.json())
        answer = response.json().get("answer", "No answer returned.")
        history.append((message, answer))
        return "", history
    else:
        error_msg = f"Error from API: {response.text}"
        history.append((message, error_msg))
        return "", history


def reset_context(session_id):
    if session_id is not None:
        reset_input = ResetChatHistoryInput(session_id=session_id)
        response = requests.post(f"{API_BASE_URL}/reset_chat_history", json=reset_input.dict())
        if response.status_code == 200:
            return "Context has been reset.", []
        else:
            return f"Failed to reset context: {response.text}", []
    return "No active session to reset.", []


theme = gr.themes.Origin(primary_hue="blue")

with gr.Blocks(theme=theme) as demo:
    session_id = gr.State(None)
    logger.debug("Demo initialized")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Document")
            url_input = gr.Textbox(label="Or enter URL")
            process_btn = gr.Button("Process")

        with gr.Column():
            output = gr.Textbox(label="Status")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    with gr.Row():
        chat_btn = gr.Button("Chat")
        reset_btn = gr.Button("Reset Context")
    logger.debug("Chatbot initialized")

    process_btn.click(
        fn=process_document,
        inputs=[file_input, url_input, session_id],
        outputs=[output, session_id],
    )
    chat_btn.click(fn=chat, inputs=[msg, chatbot, session_id], outputs=[msg, chatbot])
    msg.submit(fn=chat, inputs=[msg, chatbot, session_id], outputs=[msg, chatbot])
    reset_btn.click(fn=reset_context, inputs=[session_id], outputs=[output, chatbot])

if __name__ == "__main__":
    demo.launch()