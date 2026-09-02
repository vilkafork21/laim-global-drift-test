"""
Обёртка над GigaChat для работы с эмбеддингами.

Батчинг, retry и mean-pooling embeddings частей длинного текста.
"""

import logging
import time
from typing import Iterable, List, Optional

import numpy as np
from gigachat import GigaChat

logger = logging.getLogger(__name__)


# Размер батча на один запрос к GigaChat /embeddings.
# Подобран эмпирически с запасом против лимита API.
DEFAULT_BATCH_SIZE = 100
DEFAULT_RETRIES = 3
DEFAULT_BASE_BACKOFF = 1.0  # секунды
DEFAULT_MAX_CHARS = 1000
DEFAULT_MIN_CHARS = 200


class GigaEmbed(GigaChat):
    """
    Расширение GigaChat для эмбеддингов с батчингом и retry.
    """

    def __init__(self, *args, batch_size: int = DEFAULT_BATCH_SIZE,
                 retries: int = DEFAULT_RETRIES, base_backoff: float = DEFAULT_BASE_BACKOFF,
                 max_chars: int = DEFAULT_MAX_CHARS, min_chars: int = DEFAULT_MIN_CHARS,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_size = batch_size
        self._retries = retries
        self._base_backoff = base_backoff
        self._max_chars = int(max_chars)
        self._min_chars = int(min_chars)

    @staticmethod
    def _is_token_limit_error(err: Exception) -> bool:
        text = str(err).lower()
        return (
            "413" in text
            or "tokens limit exceeded" in text
            or "payload too large" in text
        )

    def get_embedding(self, list_of_texts: Iterable[str]) -> np.ndarray:
        """
        Получение эмбеддингов для списка текстов с автоматическим батчингом
        и retry. Возвращает numpy.ndarray формы (n, dim).
        """
        texts = ["" if text is None else str(text) for text in list_of_texts]
        if not texts:
            return np.zeros((0, 0), dtype=float)

        chunked = sum(len(text) > self._max_chars for text in texts)
        if chunked:
            logger.warning(
                "GigaEmbed: %d из %d текстов длиннее %d символов — разбиты на части",
                chunked,
                len(texts),
                self._max_chars,
            )

        chunks = []
        spans = []
        for text in texts:
            start = len(chunks)
            parts = [
                text[offset : offset + self._max_chars]
                for offset in range(0, len(text), self._max_chars)
            ]
            chunks.extend(parts or [""])
            spans.append((start, len(chunks)))

        all_embeddings: List[List[float]] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start:start + self._batch_size]
            embs = self._embed_batch(batch)
            all_embeddings.extend(embs)
        if len(all_embeddings) != len(chunks):
            raise RuntimeError(
                f"GigaEmbed: получено {len(all_embeddings)} embeddings для {len(chunks)} частей"
            )
        vectors = np.asarray(all_embeddings, dtype=float)
        return np.asarray(
            [vectors[start:end].mean(axis=0) for start, end in spans],
            dtype=float,
        )

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        last_err: Optional[Exception] = None
        net_attempts = 0
        while True:
            try:
                response = self.embeddings(batch)
                return [item.embedding for item in response.data]
            except Exception as err:
                last_err = err
                if self._is_token_limit_error(err) and any(
                    len(text) > self._min_chars for text in batch
                ):
                    logger.warning(
                        "GigaEmbed: лимит токенов — разбиваю %d текстов на части",
                        len(batch),
                    )
                    pooled = []
                    for text in batch:
                        if len(text) <= self._min_chars:
                            pooled.extend(self._embed_batch([text]))
                            continue
                        middle = len(text) // 2
                        vectors = self._embed_batch([text[:middle], text[middle:]])
                        pooled.append(np.asarray(vectors, dtype=float).mean(axis=0).tolist())
                    return pooled
                net_attempts += 1
                if net_attempts >= self._retries:
                    raise RuntimeError(
                        f"GigaEmbed: исчерпаны попытки ({self._retries})"
                    ) from last_err
                wait = self._base_backoff * (2 ** (net_attempts - 1))
                logger.warning(
                    "GigaEmbed batch failed (attempt %d/%d): %s; повтор через %.1fs",
                    net_attempts,
                    self._retries,
                    err,
                    wait,
                )
                time.sleep(wait)
