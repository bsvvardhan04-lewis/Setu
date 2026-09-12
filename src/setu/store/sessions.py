"""Durable storage for consultations.

Until now a session lived only in memory: restart the app and every visit was gone, and a
take-home card URL handed to a patient stopped working the moment the clinic rebooted. For
a tool a clinic would actually use, that is not a limitation, it is a defect.

**On privacy.** Persisting a consultation means protected health information now sits on
the disk. The product's promise is that nothing leaves the device, and that stays true —
but "does not leave" is not the same as "is not kept", so this module is deliberate about
the difference:

* everything lives in one sqlite file under the user's own data directory, which they can
  delete with the file manager and no special tooling
* a session can be deleted outright, and deletion cascades to every turn, plan item and
  gloss rather than leaving orphans behind
* retention is bounded — `prune` drops anything older than the configured window, so a
  clinic that never thinks about it does not accumulate years of consultations by default

Writes are whole-session replacements rather than incremental patches. A consultation is
tens of rows, the cost is irrelevant, and it removes an entire class of
partially-written-state bugs that would be very unpleasant in this particular domain.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consultations (
    session_id       TEXT PRIMARY KEY,
    domain_key       TEXT NOT NULL,
    patient_language TEXT NOT NULL,
    doctor_language  TEXT NOT NULL DEFAULT 'en',
    started_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS consult_turns (
    session_id TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    speaker    TEXT NOT NULL,
    text       TEXT NOT NULL,
    language   TEXT,
    at         REAL,
    jargon     TEXT DEFAULT '[]',
    PRIMARY KEY (session_id, ordinal)
);
CREATE TABLE IF NOT EXISTS consult_plan (
    session_id  TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    source_turn INTEGER NOT NULL DEFAULT -1,
    confirmed   INTEGER NOT NULL DEFAULT 0,
    similarity  REAL,
    lexical     REAL,
    semantic    REAL,
    PRIMARY KEY (session_id, ordinal)
);
CREATE TABLE IF NOT EXISTS consult_glosses (
    session_id TEXT NOT NULL,
    term       TEXT NOT NULL,
    plain      TEXT NOT NULL,
    PRIMARY KEY (session_id, term)
);
CREATE INDEX IF NOT EXISTS consultations_updated ON consultations(updated_at DESC);
"""


class SessionStore:
    """Consultations on disk. One instance per process; safe across threads."""

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

    def save(self, session) -> None:
        """Persist a whole session, replacing whatever was stored for it.

        Whole-session replacement rather than incremental patching: a consultation is tens
        of rows, so the cost is irrelevant, and it rules out half-written state - which in
        this domain could mean a care plan missing its red-flag line.
        """
        now = time.time()
        turns = [
            (
                session.session_id,
                i,
                t.speaker,
                t.text,
                t.language,
                t.at,
                json.dumps(t.jargon),
            )
            for i, t in enumerate(session.turns)
        ]
        plan = [
            (
                session.session_id,
                i,
                p.kind,
                p.text,
                p.source_turn,
                1 if p.confirmed else 0,
                p.similarity,
                p.lexical,
                p.semantic,
            )
            for i, p in enumerate(session.plan)
        ]
        glosses = [(session.session_id, term, plain) for term, plain in session.glosses.items()]

        with self._lock:
            with self._conn:  # one transaction; all of it lands or none of it does
                self._conn.execute(
                    "INSERT INTO consultations"
                    " (session_id, domain_key, patient_language, doctor_language,"
                    "  started_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(session_id) DO UPDATE SET"
                    "  domain_key=excluded.domain_key,"
                    "  patient_language=excluded.patient_language,"
                    "  updated_at=excluded.updated_at",
                    (
                        session.session_id,
                        session.domain_key,
                        session.patient_language,
                        session.doctor_language,
                        session.started_at,
                        now,
                    ),
                )
                for table in ("consult_turns", "consult_plan", "consult_glosses"):
                    self._conn.execute(
                        f"DELETE FROM {table} WHERE session_id = ?", (session.session_id,)
                    )
                if turns:
                    self._conn.executemany(
                        "INSERT INTO consult_turns"
                        " (session_id, ordinal, speaker, text, language, at, jargon)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        turns,
                    )
                if plan:
                    self._conn.executemany(
                        "INSERT INTO consult_plan"
                        " (session_id, ordinal, kind, text, source_turn, confirmed,"
                        "  similarity, lexical, semantic)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        plan,
                    )
                if glosses:
                    self._conn.executemany(
                        "INSERT INTO consult_glosses (session_id, term, plain)"
                        " VALUES (?, ?, ?)",
                        glosses,
                    )

    # ------------------------------------------------------------------- reading

    def load(self, session_id: str, session_factory):
        """Rebuild a session, or return None if it was never stored.

        `session_factory` builds the empty session so this module never has to import the
        pipeline - the store stays a store.
        """
        with self._lock:
            head = self._conn.execute(
                "SELECT * FROM consultations WHERE session_id = ?", (session_id,)
            ).fetchone()
            if head is None:
                return None
            turns = self._conn.execute(
                "SELECT * FROM consult_turns WHERE session_id = ? ORDER BY ordinal",
                (session_id,),
            ).fetchall()
            plan = self._conn.execute(
                "SELECT * FROM consult_plan WHERE session_id = ? ORDER BY ordinal",
                (session_id,),
            ).fetchall()
            glosses = self._conn.execute(
                "SELECT term, plain FROM consult_glosses WHERE session_id = ?",
                (session_id,),
            ).fetchall()

        return session_factory(dict(head), [dict(r) for r in turns], [dict(r) for r in plan],
                               {r["term"]: r["plain"] for r in glosses})

    def recent(self, limit: int = 25) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.session_id, c.domain_key, c.patient_language, c.started_at,"
                "       c.updated_at,"
                "       (SELECT COUNT(*) FROM consult_turns t"
                "         WHERE t.session_id = c.session_id) AS turns,"
                "       (SELECT COUNT(*) FROM consult_plan p"
                "         WHERE p.session_id = c.session_id) AS plan_items"
                " FROM consultations c ORDER BY c.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ deleting

    def delete(self, session_id: str) -> bool:
        """Remove a consultation and everything belonging to it."""
        with self._lock:
            with self._conn:
                for table in ("consult_turns", "consult_plan", "consult_glosses"):
                    self._conn.execute(
                        f"DELETE FROM {table} WHERE session_id = ?", (session_id,)
                    )
                cur = self._conn.execute(
                    "DELETE FROM consultations WHERE session_id = ?", (session_id,)
                )
        return cur.rowcount > 0

    def prune(self, older_than_days: int) -> int:
        """Drop consultations past the retention window.

        Bounded retention is the default rather than an option, so a clinic that never
        thinks about this does not silently accumulate years of patient conversations.
        """
        if older_than_days <= 0:
            return 0
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            stale = [
                r["session_id"]
                for r in self._conn.execute(
                    "SELECT session_id FROM consultations WHERE updated_at < ?", (cutoff,)
                ).fetchall()
            ]
        for session_id in stale:
            self.delete(session_id)
        return len(stale)

    def stats(self) -> dict[str, int]:
        with self._lock:
            sessions = self._conn.execute("SELECT COUNT(*) FROM consultations").fetchone()[0]
            turns = self._conn.execute("SELECT COUNT(*) FROM consult_turns").fetchone()[0]
        return {"consultations": int(sessions), "turns": int(turns)}
