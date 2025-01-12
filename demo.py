import gradio as gr
import requests
import uuid
from typing import Optional
from fastapi import UploadFile
from pydantic_models import DocumentInput
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


session_manager = SessionManager()


def process_document(file_obj: UploadFile, session_id):
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
                "Status code: {response.status_code}\n"
            )

    return response_msg, session_id


def chat(message, history, session_id):
    logger.debug('New message in chat: {}', message)
    logger.debug('Chat history: {}', history)
    logger.debug('Session ID: {}', session_id)

    if session_id is None:
        error_msg = "No active session. Please upload a document first."
        history.append((message, error_msg))
        return "", history

    response = requests.post(
        f"{API_BASE_URL}/chat", json={
            "question": message,
            "session_id": session_id
            }
    )

    if response.status_code == 200:
        logger.debug('Successful response: {}', response.json())
        answer = response.json().get("answer", "No answer returned.")
        history.append((message, answer))
        return "", history
    else:
        error_msg = f"Error from API: {response.text}"
        history.append((message, error_msg))
        return "", history


with gr.Blocks() as demo:
    session_id = gr.State(None)
    logger.debug('Demo initialized')

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Document")
            process_btn = gr.Button("Process")

        with gr.Column():
            output = gr.Textbox(label="Status")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")
    logger.debug('Chatbot initialized')

    process_btn.click(
        fn=process_document,
        inputs=[file_input, session_id],
        outputs=[output, session_id],
    )
    msg.submit(fn=chat,
               inputs=[msg, chatbot, session_id],
               outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.launch()
