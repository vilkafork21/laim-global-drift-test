from __future__ import annotations

from types import SimpleNamespace

from giga_wraper import GigaEmbed


class _LimitOnceGigaEmbed(GigaEmbed):
    def __init__(self):
        self._batch_size = 100
        self._retries = 1
        self._base_backoff = 0
        self._max_chars = 1000
        self._min_chars = 200
        self.calls = []

    def embeddings(self, batch):
        self.calls.append([len(text) for text in batch])
        if len(self.calls) == 1:
            raise RuntimeError("413 Tokens limit exceeded")
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0]) for _text in batch]
        )


class _RecordingGigaEmbed(GigaEmbed):
    def __init__(self):
        self._batch_size = 100
        self._retries = 1
        self._base_backoff = 0
        self._max_chars = 4
        self._min_chars = 1
        self.calls = []

    def embeddings(self, batch):
        self.calls.extend(batch)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[len(text), float(text == "TAIL")])
                for text in batch
            ]
        )


def test_token_limit_splits_text_without_changing_output_count():
    model = _LimitOnceGigaEmbed()

    embeddings = model.get_embedding(["x" * 1200])

    assert embeddings.tolist() == [[1.0]]
    assert model.calls == [[1000, 200], [500, 500], [200]]


def test_long_dialogue_embeds_tail_and_keeps_one_vector_per_input():
    model = _RecordingGigaEmbed()

    embeddings = model.get_embedding(["HEADTAIL", "one"])

    assert model.calls == ["HEAD", "TAIL", "one"]
    assert embeddings.tolist() == [[4.0, 0.5], [3.0, 0.0]]
