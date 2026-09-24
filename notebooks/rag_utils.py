"""Shared retrieval helpers for the RAG notebooks (05, 06, 11).

Notebook 04 builds every piece below step by step; later notebooks import
them from here so they can focus on their own topic.
"""
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# The same eight-document pet knowledge base used in notebook 04
PET_CORPUS: List[Dict[str, str]] = [
    {"id": "d1", "text": "The Siberian Husky is a medium-sized working dog breed originally bred for sled pulling in cold climates."},
    {"id": "d2", "text": "Persian cats have long, thick fur and a calm, gentle temperament, making them popular indoor pets."},
    {"id": "d3", "text": "Goldfish are freshwater fish commonly kept in home aquariums; they can live over 10 years with proper care."},
    {"id": "d4", "text": "African Grey Parrots are known for their exceptional ability to mimic human speech and solve problems."},
    {"id": "d5", "text": "Bengal cats are a hybrid breed resulting from crossing domestic cats with the Asian leopard cat."},
    {"id": "d6", "text": "Golden Retrievers are friendly, intelligent dogs often used as guide dogs and family companions."},
    {"id": "d7", "text": "Bearded dragons are docile reptiles native to Australia, popular as low-maintenance terrarium pets."},
    {"id": "d8", "text": "Canaries are small songbirds prized for their bright colors and melodic singing ability."},
]

_embed_tokenizer = None
_embed_model = None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_texts(texts: List[str]) -> np.ndarray:
    """Mean-pool token embeddings (masking out padding) into one L2-normalized vector per text."""
    global _embed_tokenizer, _embed_model
    if _embed_model is None:
        _embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL_ID)
        _embed_model.eval()
    encoded = _embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        token_embeddings = _embed_model(**encoded)[0]
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """Fuse several ranked doc_id lists into one, scoring by 1/(k + rank) summed across rankings."""
    fused_scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """BM25 + embedding search over a list of {"id", "text"} documents, fused with RRF."""

    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.by_id: Dict[str, str] = {doc["id"]: doc["text"] for doc in corpus}
        self.doc_ids = [doc["id"] for doc in corpus]
        self.bm25 = BM25Okapi([_tokenize(doc["text"]) for doc in corpus])
        self.embeddings = embed_texts([doc["text"] for doc in corpus])

    def lexical_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def vector_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        similarities = self.embeddings @ embed_texts([query])[0]
        ranked = sorted(zip(self.doc_ids, similarities), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def hybrid_search(self, query: str, top_k: Optional[int] = 3) -> List[Tuple[str, float]]:
        lexical_ranking = [doc_id for doc_id, _ in self.lexical_search(query)]
        vector_ranking = [doc_id for doc_id, _ in self.vector_search(query)]
        return reciprocal_rank_fusion([lexical_ranking, vector_ranking])[:top_k]
