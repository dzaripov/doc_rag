import os

from dotenv import load_dotenv
from typing import List, Dict
from langchain import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_milvus import Milvus
from hydra import initialize, compose
from mistral import MistralLLM, MistralEmbed
from retriever import retrieve_chunks
from reranker import rerank_chunks


class RAGPipeline:
    def __init__(self, config_path: str, texts: List[str]):
        # Load Hydra configuration
        with initialize(config_path=config_path):
            self.config = compose(config_name="config")

        load_dotenv('../.env')
        MISTRAL_API_KEY = os.getenv(self.config['llm']['env_api_key'])
        self.llm = MistralLLM(api_key=MISTRAL_API_KEY,
                              model_name=self.config['llm']['model_name'],
                              api_url=self.config['llm']['api_url'])
        self.embeddings = MistralEmbed(texts)
        self.vectorstore = Milvus(
                    self.embeddings,
                    # ADD NEEDED PARAMS
                    # connection_args={"uri": f"./{self.vectorstore_path}.db"},
                    # collection_name="RAG",
                )

    def setup_qa_chain(self, custom_prompt: str = None):
        retrieve_results = retrieve_chunks(cfg=self.config, query=custom_prompt, vectorstore=self.vectorstore)
        rerank_results = rerank_chunks(cfg=self.config, query=custom_prompt, chunks=retrieve_results)
        # IF DO IT THIS WAY, HERE NEED TO PREPARE A REQUEST TO THE LLM

        system_template = "You are an assistant that helps user with documentation. \
        Based on the following information, give an answer to the user's request: \
        {search_results}"
        user_template = "User's request was: {input_text}"

        system_prompt_template = PromptTemplate(input_variables=["search_results"], template=system_template)
        user_prompt_template = PromptTemplate(input_variables=["input_text"], template=user_template)

        formatted_system_prompt = system_prompt_template.format(search_results='\n'.join(rerank_results))
        formatted_user_prompt = user_prompt_template.format(input_text=custom_prompt)

        answer = self.llm.generate(formatted_system_prompt, formatted_user_prompt)
        return answer

    def query(self, question: str) -> Dict:
        # Run the QA chain with the provided question
        return self.setup_qa_chain(question)
