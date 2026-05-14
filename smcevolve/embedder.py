"""Async embedding client with caching for program diversity selection.

Uses an OpenAI-compatible /embeddings endpoint. Caches results by
program content hash to avoid redundant API calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    """Async embedding client with an in-memory cache."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
        )
        self._lock = asyncio.Lock()
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Return L2-normalised embeddings, using cache where possible."""
        results: list[np.ndarray | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            k = self._key(t)
            cached = self._cache.get(k)
            if cached is not None:
                results[i] = cached
            else:
                to_fetch.append((i, t))

        if to_fetch:
            uncached_texts = [t for _, t in to_fetch]
            embeddings = await self._fetch(uncached_texts)
            async with self._lock:
                for (idx, t), emb in zip(to_fetch, embeddings):
                    self._cache[self._key(t)] = emb
                    results[idx] = emb

        return results  # type: ignore[return-value]

    async def _fetch(self, texts: list[str]) -> list[np.ndarray]:
        """Call the embedding API and return L2-normalised vectors."""
        try:
            resp = await self._client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format="float",
            )
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            vecs = [
                np.array(d.embedding, dtype=np.float32)
                for d in sorted_data
            ]
            # L2-normalise so downstream can use plain L2 distance
            for i, v in enumerate(vecs):
                norm = np.linalg.norm(v)
                if norm > 1e-12:
                    vecs[i] = v / norm
            return vecs
        except Exception as exc:
            log.warning("Embedding call failed: %s", exc)
            return [np.zeros(256, dtype=np.float32) for _ in texts]
