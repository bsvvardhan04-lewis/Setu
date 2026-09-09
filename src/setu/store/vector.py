"""Local document + vector store.

sqlite plus numpy brute-force cosine. No server, no index build, no network. For a
personal corpus of a few thousand chunks this is faster than any ANN index once you
count the build cost, and it has the property that matters most here: it is a single
file on the user's disk that they can delete.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    language TEXT,
    created_at REAL DEFAULT (strftime('%s','now')),
    meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL DEFAULT 0,
    ordinal INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    box TEXT DEFAULT '[]',
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(document_id);
"""


@dataclass
class Chunk:
    id: int
    document_id: int
    page: int
    ordinal: int
    text: str
    box: list[int]
    score: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page": self.page,
            "ordinal": self.ordinal,
            "text": self.text,
            "box": self.box,
            "score": round(self.score, 4),
        }


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking that never splits mid-sentence when it can help it."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= size:
            buffer = f"{buffer}\n{para}" if buffer else para
            continue
        if buffer:
            chunks.append(buffer)
        if len(para) <= size:
            buffer = para
        else:
            step = max(1, size - overlap)
            for i in range(0, len(para), step):
                piece = para[i : i + size]
                if piece:
                    chunks.append(piece)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


class VectorStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------- writing

    def add_document(self, title: str, language: str = "en", meta: dict | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO documents (title, language, meta) VALUES (?, ?, ?)",
                (title, language, json.dumps(meta or {})),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_chunks(
        self,
        document_id: int,
        texts: list[str],
        embeddings: np.ndarray,
        pages: list[int] | None = None,
        boxes: list[list[int]] | None = None,
    ) -> int:
        if not texts:
            return 0
        pages = pages or [0] * len(texts)
        boxes = boxes or [[] for _ in texts]
        rows = [
            (
                document_id,
                int(pages[i]),
                i,
                texts[i],
                json.dumps(boxes[i]),
                np.asarray(embeddings[i], dtype=np.float32).tobytes(),
            )
            for i in range(len(texts))
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO chunks (document_id, page, ordinal, text, box, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def delete_document(self, document_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            self._conn.commit()

    # ------------------------------------------------------------------- reading

    def documents(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.id, d.title, d.language, d.created_at, COUNT(c.id) AS chunks"
                " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
                " GROUP BY d.id ORDER BY d.id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def search(
        self, query_vec: np.ndarray, top_k: int = 6, document_id: int | None = None
    ) -> list[Chunk]:
        sql = "SELECT id, document_id, page, ordinal, text, box, embedding FROM chunks"
        args: tuple = ()
        if document_id is not None:
            sql += " WHERE document_id = ?"
            args = (document_id,)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        if not rows:
            return []

        matrix = np.stack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        )
        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        norms = np.clip(np.linalg.norm(matrix, axis=1) * np.linalg.norm(query), 1e-9, None)
        scores = (matrix @ query) / norms

        order = np.argsort(-scores)[:top_k]
        return [
            Chunk(
                id=rows[i]["id"],
                document_id=rows[i]["document_id"],
                page=rows[i]["page"],
                ordinal=rows[i]["ordinal"],
                text=rows[i]["text"],
                box=json.loads(rows[i]["box"] or "[]"),
                score=float(scores[i]),
            )
            for i in order
        ]

    def stats(self) -> dict[str, int]:
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": int(docs), "chunks": int(chunks)}
