from omegaconf import DictConfig, OmegaConf
from typing import List


def retrieve_chunks(cfg: DictConfig, query: str, vectorstore):
    # SHOULD ADD HERE SELF-RAG OR OTHER RETRIEVE HACKS
    retriever = vectorstore.as_retriever(search_kwargs=OmegaConf.to_container(cfg.retriever))
    docs = retriever.invoke(query)
    return docs