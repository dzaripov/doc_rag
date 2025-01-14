import os
import logging

from dotenv import load_dotenv
from typing import Dict
from langchain_core.prompts import PromptTemplate

from .mistral import MistralLLM

from .retriever import retrieve_chunks
from .reranker import rerank_chunks


class RAGPipeline:
    def __init__(self):
        load_dotenv(".env")
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

        self.llm = MistralLLM(
            api_key=MISTRAL_API_KEY,
            model_name="mistral-large-latest",
            api_url="https://api.mistral.ai/v1/",
        )
        self.document_stores = {}

    def setup_qa_chain(self, question: str, chat_history: str, session_id: str):
        logging.debug(f"Setting up QA chain for session {session_id}")
        logging.debug(f"Available document stores: {list(self.document_stores.keys())}")

        if session_id not in self.document_stores:
            raise ValueError(f"No document store found for session {session_id}")

        cfg = {"retriever": "vectorstore", "reranker": "bm25"}

        retrieve_results = retrieve_chunks(
            cfg, query=question, store=self.document_stores[session_id]
        )

        retrieve_texts = [doc.page_content for doc in retrieve_results]

        rerank_results = rerank_chunks(cfg=cfg, query=question, chunks=retrieve_texts)

        system_template = """You are an assistant dedicated to helping users with documentation. 

Your task is to provide answers based on the information retrieved from search results. Follow these guidelines:

1. **Accuracy**: If the search results contain sufficient information to answer the user's question, provide a clear and accurate response.
2. **Transparency**: If the search results do not contain enough information to answer the question, honestly state that you don't know rather than making up an answer.
3. **Language Consistency**: Always respond in the same language in which the question was asked, even if the documentation is in another language.
4. **Avoid Speculation**: Do not fabricate or infer information that is not directly supported by the search results.
5. **Contextual Understanding**: Ensure you understand the full context of the user's question before responding. Pay attention to any specific details or nuances.
6. **Complex Questions**: For multi-part questions or complex queries, break down your response to address each part individually.
7. **Stay on Topic**: Only answer questions that are directly related to the search results. If a question is unrelated or off-topic, politely inform the user that you can only address questions relevant to the original topic.

By adhering to these principles, you ensure that users receive reliable and helpful responses."""
        user_template = (
            "Request: {input_text}\n"
            "Search results for this request: {search_results}\n"
            "Conversation history:\n{chat_history}"
        )

        system_prompt_template = PromptTemplate(
            input_variables=[], template=system_template
        )
        user_prompt_template = PromptTemplate(
            input_variables=["input_text", "search_results", "chat_history"],
            template=user_template,
        )

        formatted_system_prompt = system_prompt_template.format()
        formatted_user_prompt = user_prompt_template.format(
            input_text=question,
            search_results="\n".join(rerank_results),
            chat_history=chat_history,
        )

        answer = self.llm.generate(formatted_system_prompt, formatted_user_prompt)
        return {"answer": answer}

    def invoke(self, question: str, chat_history: str, session_id) -> Dict:
        return self.setup_qa_chain(question, chat_history, session_id)
