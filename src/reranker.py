# import numpy as np
import hydra
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from omegaconf import DictConfig
from typing import List, Tuple
from loguru import logger
from retriever import retrieve_bm25 as rerank_bm25
# from catboost import CatBoostRanker, Pool


def rerank_cross_encoder(cfg: DictConfig, query: str, chunks: List[str]):
    model_name = cfg['reranker']['model_name']
    top_k = cfg['reranker']['top_k']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg['reranker']['model_name'])

    inputs = tokenizer(
        [[query, chunk] for chunk in chunks],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[:, 0]  # Assuming binary classification: relevance score is the first logit

    # Combine scores with corresponding chunks
    chunk_scores = [(chunk, score.item()) for chunk, score in zip(chunks, scores)]
    # Sort by scores in descending order
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)

    return sorted_chunks[:top_k]



def rerank_chunks(cfg: DictConfig, query: str, chunks: List[str]):
    try:
        reranker_type = cfg['reranker']
        if reranker_type == 'bm25':
            sorted_chunks = rerank_bm25(cfg, query, chunks)
        elif reranker_type == 'cross_encoder':
            sorted_chunks = rerank_cross_encoder(cfg, query, chunks)
        else:
            raise ValueError(f'Unknown ranking type: {reranker_type}')

        return sorted_chunks

    except Exception as e:
        logger.error("Error: {}", e)
        raise
