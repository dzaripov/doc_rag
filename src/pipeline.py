from typing import List, Dict
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

        self.llm = MistralLLM
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
        answer = self.llm(system_prompt, user_prompt)
        return answer

    def query(self, question: str) -> Dict:
        # Run the QA chain with the provided question
        return self.setup_qa_chain(question)
