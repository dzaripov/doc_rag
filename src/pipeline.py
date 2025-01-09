import os
import logging

from dotenv import load_dotenv
from typing import List, Dict
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
        self.document_stores = {}  # different document stores for sessions

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

        # rerank_results = rerank_chunks(
        #     cfg=cfg,
        #     query=question,
        #     chunks=retrieve_texts
        # )
        rerank_results = retrieve_texts

        system_template = (
            "You are an assistant that helps user with documentation.\n"
            "Based on the information search results, give an answer to the user's request:\n"
            "If you can't give the answer based only on search results, say that you don't know, don't make it up yourself."
            "Also, if the found information doesn't answer the question, don't make it up, but say there is not enough information."
            "Answer in the same language in which you were asked the question."
            "If the documentation is in a different language, still use the language of the question."
        )
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
        return {
            "answer": answer
        }

    def invoke(self, question: str, chat_history: str, session_id) -> Dict:
        # Run the QA chain with the provided question
        return self.setup_qa_chain(question, chat_history, session_id)
