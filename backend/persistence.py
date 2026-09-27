"""Call-event persistence: SQLite store for every WS event a call emits.

Per BACKEND_COMPLETION.md Sec3.6: nothing survives a disconnect today -
main.py sends a rich stream of structured events (vad_start, transcript,
agent_clause, ...) straight to the browser and nowhere else, so there is no
call history for a dashboard/Call Log to read once the socket closes. This
gives every event a durable home, keyed by connection_id, without touching
the events themselves - main.py's send_event() gains one extra write, the
event shape it already builds is exactly what gets stored.

SQLite (stdlib, one file, no server) is enough for v1's single-process
deployment - see PersistenceSettings. sqlite3 connections aren't thread/task
-safe for concurrent writers, so writes go through asyncio.to_thread() with
one connection per call to CallEventStore.record(), mirroring how asr.py/
tts.py already push blocking model calls off the event loop.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time

from cryptography.fernet import Fernet, InvalidToken

from .settings import PersistenceSettings

logger = logging.getLogger("aica.persistence")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    connection_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL
);

CREATE TABLE IF NOT EXISTS call_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    connection_id TEXT NOT NULL,
    ts REAL NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_call_events_connection_id ON call_events(connection_id);
"""


class CallEventStore:
    """Owns one SQLite file and records every event/call lifecycle transition.

    Never raises out of record()/start_call()/end_call(): persistence is a
    side channel for history/dashboards, not something that should ever take
    down a live call if the disk is full or the file is locked.
    """

    def __init__(self, settings: PersistenceSettings) -> None:
        self.settings = settings

        # Encryption-at-rest for event payloads (BACKEND_COMPLETION.md Sec4).
        # Validated eagerly here - not lazily inside record() - so a
        # misconfigured key fails loudly at startup instead of silently
        # writing corrupt/unrecoverable rows later. Generate a key with:
        #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
        self._fernet: Fernet | None = None
        if self.settings.encryption_key:
            try:
                self._fernet = Fernet(self.settings.encryption_key.encode("utf-8"))
            except Exception as exc:
                raise ValueError(
                    "CALL_EVENTS_ENCRYPTION_KEY is set but is not a valid Fernet key "
                    "(must be 32 url-safe base64-encoded bytes). Generate one with "
                    '`python -c "from cryptography.fernet import Fernet; '
                    'print(Fernet.generate_key().decode())"`'
                ) from exc

    def load(self) -> None:
        """Create the DB file and schema once during startup."""
        connection = sqlite3.connect(self.settings.db_path)
        try:
            connection.executescript(_SCHEMA)
            connection.commit()
        finally:
            connection.close()
        logger.info("call-event store ready at %s", self.settings.db_path)

    def start_call(self, connection_id: str) -> None:
        self._execute(
            "INSERT OR REPLACE INTO calls (connection_id, started_at, ended_at) VALUES (?, ?, NULL)",
            (connection_id, time.time()),
        )

    def end_call(self, connection_id: str) -> None:
        self._execute(
            "UPDATE calls SET ended_at = ? WHERE connection_id = ?",
            (time.time(), connection_id),
        )

    def record(self, connection_id: str, event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        if self._fernet is not None:
            payload = self._fernet.encrypt(payload.encode("utf-8")).decode("ascii")
        self._execute(
            "INSERT INTO call_events (connection_id, ts, event_type, payload) VALUES (?, ?, ?, ?)",
            (connection_id, time.time(), str(event.get("type")), payload),
        )

    def recent_calls(self, limit: int = 50) -> list[dict]:
        """List calls newest-first, with a cheap per-call summary.

        This is what turns the event log into a Call Log a dashboard can render
        without downloading every call's full event stream (BACKEND_COMPLETION.md
        Sec3.6). The counts and timings come from SQL aggregates rather than
        from decrypting payloads, so this stays fast - and keeps working when
        encryption is on, since event_type is deliberately stored alongside the
        encrypted payload rather than inside it.
        """
        connection = sqlite3.connect(self.settings.db_path)
        try:
            rows = connection.execute(
                """
                SELECT c.connection_id, c.started_at, c.ended_at,
                       COUNT(e.id),
                       SUM(CASE WHEN e.event_type = 'transcript' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN e.event_type = 'agent_clause' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN e.event_type = 'agent_tool_call' THEN 1 ELSE 0 END)
                FROM calls c
                LEFT JOIN call_events e ON e.connection_id = c.connection_id
                GROUP BY c.connection_id
                ORDER BY c.started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        except Exception:
            logger.exception("call-event history read failed")
            return []
        finally:
            connection.close()

        return [
            {
                "connection_id": connection_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": round(ended_at - started_at, 3) if ended_at else None,
                "event_count": event_count or 0,
                "caller_turns": caller_turns or 0,
                "agent_clauses": agent_clauses or 0,
                "tool_calls": tool_calls or 0,
            }
            for connection_id, started_at, ended_at, event_count, caller_turns, agent_clauses, tool_calls in rows
        ]

    def events_for_call(self, connection_id: str) -> list[dict]:
        """Read back one call's full event history, oldest first. For dashboards/tests, not the hot path."""
        connection = sqlite3.connect(self.settings.db_path)
        try:
            rows = connection.execute(
                "SELECT event_type, payload FROM call_events WHERE connection_id = ? ORDER BY id ASC",
                (connection_id,),
            ).fetchall()
        finally:
            connection.close()

        events = []
        for _event_type, payload in rows:
            plaintext = payload
            if self._fernet is not None:
                try:
                    plaintext = self._fernet.decrypt(payload.encode("utf-8")).decode("utf-8")
                except InvalidToken:
                    # Row was encrypted with a different/rotated key, or was
                    # never encrypted at all - drop it rather than crash the
                    # whole read, matching this module's "persistence errors
                    # should never take down the caller" philosophy.
                    logger.error(
                        "call-event payload for connection_id=%s could not be decrypted "
                        "(wrong/rotated key, or row predates encryption) - skipping",
                        connection_id,
                    )
                    continue
            try:
                events.append(json.loads(plaintext))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.exception("call-event payload for connection_id=%s is not valid JSON - skipping", connection_id)
                continue
        return events

    def _execute(self, query: str, params: tuple) -> None:
        try:
            connection = sqlite3.connect(self.settings.db_path)
            try:
                connection.execute(query, params)
                connection.commit()
            finally:
                connection.close()
        except Exception:
            logger.exception("call-event persistence write failed (query=%s)", query.split()[0])
