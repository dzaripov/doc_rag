import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from loguru import logger
from rank_bm25 import BM25Okapi

from .utils import tokenize_text


def rerank_bm25(query: str, chunks: List[str]):

    tokenized_query = tokenize_text(query)
    logger.debug("Query prepared succesfully: {}", tokenized_query)

    corpus = [tokenize_text(chunk) for chunk in chunks]
    logger.debug("Corpus prepared succesfully: {}", corpus)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)

    sorted_chunks_scores = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
        )
    sorted_chunks = [chunk for chunk, _ in sorted_chunks_scores]
    return sorted_chunks


def rerank_cross_encoder(query: str, chunks: List[str]):
    model_name = "distilbert-base-uncased"
    top_k = 4
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(
        [[query, chunk] for chunk in chunks],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[:, 0]
        # Assuming binary classification: relevance score is the first logit

    chunk_scores = [(chunk, score.item()) for chunk, score in zip(chunks, scores)]
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    sorted_chunks = [chunk for chunk, score in sorted_chunks]

    return sorted_chunks[:top_k]


def rerank_chunks(cfg, query: str, chunks: List[str]):
    try:
        reranker_type = cfg["reranker"]
        if reranker_type == "bm25":
            sorted_chunks = rerank_bm25(query, chunks)
        elif reranker_type == "cross_encoder":
            sorted_chunks = rerank_cross_encoder(query, chunks)
        else:
            raise ValueError(f"Unknown ranking type: {reranker_type}")

        return sorted_chunks

    except Exception as e:
        logger.error("Error: {}", e)
        raise
