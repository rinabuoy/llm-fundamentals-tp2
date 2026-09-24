"""Shared retrieval helpers for the RAG notebooks (02 and 04).

Notebook 02 builds vector search step by step; the pipeline and agent
sections import it from here so they can focus on their own topic.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# The same eight-document pet knowledge base used in notebook 02
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


class VectorRetriever:
    """Embedding search over a list of {"id", "text"} documents, ranked by cosine similarity."""

    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.by_id: Dict[str, str] = {doc["id"]: doc["text"] for doc in corpus}
        self.doc_ids = [doc["id"] for doc in corpus]
        self.embeddings = embed_texts([doc["text"] for doc in corpus])  # embed the corpus once

    def vector_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        # embed_texts L2-normalizes, so the dot product IS the cosine similarity
        similarities = self.embeddings @ embed_texts([query])[0]
        ranked = sorted(zip(self.doc_ids, similarities), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
