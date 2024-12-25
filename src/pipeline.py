import os

from dotenv import load_dotenv
from typing import List, Dict
from langchain import PromptTemplate

from hydra import initialize, compose
from mistral import MistralLLM

from retriever import retrieve_chunks
from reranker import rerank_chunks


class RAGPipeline:
    def __init__(self, config_path: str):
        # Load Hydra configuration
        with initialize(config_path=config_path):
            self.config = compose(config_name="config")

        load_dotenv('.env')
        MISTRAL_API_KEY = os.getenv(self.config['llm']['env_api_key'])
        self.llm = MistralLLM(api_key=MISTRAL_API_KEY,
                              model_name=self.config['llm']['model_name'],
                              api_url=self.config['llm']['api_url'])
        self.document_stores = {} # different document stores for sessions

    def setup_qa_chain(self, question: str, chat_history: str):
        retrieve_results = retrieve_chunks(cfg=self.config, query=question, vectorstore=self.vectorstore)
        rerank_results = rerank_chunks(cfg=self.config, query=question, chunks=retrieve_results)

        system_template = (
            "You are an assistant that helps user with documentation.\n"
            "Based on the information search results, give an answer to the user's request:\n"
            "If you can't give the answer based only on search results, say that you don't know, don't make it up yourself."
        )
        user_template = (
            "Request: {input_text}\n"
            "Search results for this request: {search_results}\n"
            "Conversation history:\n{chat_history}"
        )

        system_prompt_template = PromptTemplate(input_variables=["search_results"], template=system_template)
        user_prompt_template = PromptTemplate(input_variables=["input_text", "chat_history"], template=user_template)

        formatted_system_prompt = system_prompt_template.format(search_results='\n'.join(rerank_results))
        formatted_user_prompt = user_prompt_template.format(input_text=question, chat_history=chat_history)

        answer = self.llm.generate(formatted_system_prompt, formatted_user_prompt)
        return answer


    def invoke(self, question: str, chat_history: str) -> Dict:
        # Run the QA chain with the provided question
        return self.setup_qa_chain(question, chat_history)
