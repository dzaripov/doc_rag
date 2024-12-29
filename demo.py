import os
import tempfile
import uuid
from typing import Optional

import gradio as gr


class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "history": [],
            "documents": [],
            "urls": []
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)

session_manager = SessionManager()

def process_document(file_obj, url, session_id):
    if session_id is None:
        session_id = session_manager.create_session()
    # не стоит ли при каждой новой загрузке документа создавать новую сессию?

    session = session_manager.get_session(session_id)
    response = ""

    if file_obj is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file_obj.read())
            session["documents"].append(tmp_file.name)
            response += f"Document processed and added to session {session_id}\n"

    if url and url.strip():
        session["urls"].append(url.strip())
        response += f"URL added to session {session_id}\n"

    return response, session_id

def chat(message, history, session_id):
    if session_id is None:
        return "Please upload a document or provide a URL first", history

    session = session_manager.get_session(session_id)
    if not session:
        return "Invalid session", history

    # Here you would implement your RAG logic using session["documents"] and session["urls"]
    response = f"Processing query: {message} for session {session_id}"

    history.append((message, response))
    return response, history

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
        outputs=[output, session_id]
    )

    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, chatbot]
    )

if __name__ == "__main__":
    demo.launch()