import gradio as gr
import requests
import uuid
import os
from typing import Optional
from fastapi import UploadFile
from pydantic_models import DocumentInput

# URL вашего API FastAPI
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


def process_document(file_obj: UploadFile, url, session_id):
    if session_id is None:
        session_id = session_manager.create_session()

    response_msg = ""
    if file_obj is not None:
        file_name = file_obj.name
        document = DocumentInput(
            docs_url=file_name,
            session_id="123456",
            config_path="custom_config",
        )
        # Serialize the DocumentInput to a dictionary
        response = requests.post(
            f"{API_BASE_URL}/upload_pdf",
            json=document.model_dump(),
        )

        if response.status_code == 200:
            response_msg += f"Document uploaded successfully to session {session_id}\n"
        else:
            response_msg += f"Failed to upload document: {response.text}. Status code: {response.status_code}\n"

    if url and url.strip():
        document = DocumentInput(
            docs_url=url,
            session_id="123456",
            config_path="custom_config",
        )
        response = requests.post(
            f"{API_BASE_URL}/upload_url",
            json=document.model_dump(),  # Use .dict() to serialize the Pydantic model
        )
        if response.status_code == 200:
            response_msg += f"URL processed successfully in session {session_id}\n"
        else:
            response_msg += f"Failed to process URL: {response.text}\n"

    return response_msg, session_id


def chat(message, history, session_id):
    if session_id is None:
        return "Please upload a document or provide a URL first", history

    # Отправка запроса в API FastAPI
    response = requests.post(
        f"{API_BASE_URL}/chat", json={"question": message, "session_id": session_id}
    )

    if response.status_code == 200:
        answer = response.json().get("answer", "No answer returned.")
        history.append((message, answer))
        return answer, history
    else:
        error_msg = f"Error from API: {response.text}"
        history.append((message, error_msg))
        return error_msg, history


with gr.Blocks() as demo:
    session_id = gr.State(None)

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Document")
            url_input = gr.Textbox(label="Or enter URL")
            process_btn = gr.Button("Process")

        with gr.Column():
            output = gr.Textbox(label="Status")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")

    process_btn.click(
        fn=process_document,
        inputs=[file_input, url_input, session_id],
        outputs=[output, session_id],
    )

    msg.submit(fn=chat, inputs=[msg, chatbot, session_id], outputs=[chatbot, chatbot])

if __name__ == "__main__":
    demo.launch()
