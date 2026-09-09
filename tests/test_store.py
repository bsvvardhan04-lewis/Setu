"""Vector store and chunking."""

from __future__ import annotations

import numpy as np
import pytest

from setu.store import VectorStore, chunk_text


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "t.db")
    yield s
    s.close()


def test_chunking_keeps_paragraphs_whole():
    text = "First para.\nSecond para.\nThird para."
    assert chunk_text(text, size=200) == [text]


def test_chunking_splits_an_oversized_paragraph():
    chunks = chunk_text("x" * 2500, size=900, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)


def test_roundtrip_and_similarity_ranking(store):
    doc = store.add_document("Notice", "hi")
    texts = ["deadline is 31 March", "office is in Bengaluru", "fee is 500 rupees"]
    vectors = np.eye(3, dtype=np.float32)
    assert store.add_chunks(doc, texts, vectors, pages=[0, 0, 1]) == 3

    hits = store.search(np.array([0, 0, 1], dtype=np.float32), top_k=2)
    assert hits[0].text == "fee is 500 rupees"
    assert hits[0].page == 1
    assert hits[0].score > hits[1].score


def test_search_can_scope_to_one_document(store):
    a = store.add_document("A")
    b = store.add_document("B")
    store.add_chunks(a, ["alpha"], np.array([[1.0, 0.0]], dtype=np.float32))
    store.add_chunks(b, ["beta"], np.array([[0.0, 1.0]], dtype=np.float32))
    hits = store.search(np.array([1.0, 1.0], dtype=np.float32), top_k=5, document_id=b)
    assert [h.text for h in hits] == ["beta"]


def test_delete_cascades(store):
    doc = store.add_document("Gone")
    store.add_chunks(doc, ["x"], np.zeros((1, 2), dtype=np.float32))
    store.delete_document(doc)
    assert store.stats() == {"documents": 0, "chunks": 0}
