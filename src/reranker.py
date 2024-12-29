# import numpy as np
import hydra
import torch
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from omegaconf import DictConfig
from typing import List, Tuple
from loguru import logger
from rank_bm25 import BM25Okapi


def rerank_bm25(query: str, chunks: List[str]):
    pattern = r"""
            [a-zA-Z_][a-zA-Z0-9_]*  # Идентификаторы (включая ключевые слова)
            | \d+\.\d+              # Числа с плавающей точкой
            | \d+                   # Целые числа
            | "([^"\\]|\\.)*"       # Строковые литералы в двойных кавычках
            | '([^'\\]|\\.)*'       # Строковые литералы в одинарных кавычках
        """
    regex = re.compile(pattern)

    tokenized_query = regex.findall(query)
    corpus = [regex.findall(chunk) for chunk in chunks]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)

    doc_scores = zip(chunks, scores)
    doc_scores.sort(key=lambda doc_score: doc_score[1], reverse=True)
    sorted_chunks = [chunk for chunk, _ in doc_scores]

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
