"""Self-check for CallEventStore: schema creation, event ordering, call lifecycle.

Uses a temp-file SQLite DB per test (sqlite3's :memory: doesn't survive across
the separate connections CallEventStore opens per call, since each connection
gets its own private in-memory DB) - no mocking needed, this is already fast
and hermetic.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from .persistence import CallEventStore
from .settings import PersistenceSettings


def _make_store(encryption_key: str = "") -> CallEventStore:
    tmp_dir = tempfile.mkdtemp()
    store = CallEventStore(
        PersistenceSettings(db_path=Path(tmp_dir) / "call_events.db", encryption_key=encryption_key)
    )
    store.load()
    return store


def test_load_is_idempotent() -> None:
    store = _make_store()
    store.load()  # must not raise or wipe existing tables on a second call


def test_record_and_read_back_events_in_order() -> None:
    store = _make_store()
    store.start_call("conn-1")
    store.record("conn-1", {"type": "vad_start", "probability": 0.9})
    store.record("conn-1", {"type": "transcript", "text": "hello"})
    store.end_call("conn-1")

    events = store.events_for_call("conn-1")

    assert [event["type"] for event in events] == ["vad_start", "transcript"]
    assert events[1]["text"] == "hello"


def test_events_for_call_only_returns_that_connections_events() -> None:
    store = _make_store()
    store.start_call("conn-1")
    store.start_call("conn-2")
    store.record("conn-1", {"type": "transcript", "text": "for conn-1"})
    store.record("conn-2", {"type": "transcript", "text": "for conn-2"})

    assert [e["text"] for e in store.events_for_call("conn-1")] == ["for conn-1"]
    assert [e["text"] for e in store.events_for_call("conn-2")] == ["for conn-2"]


def test_events_for_call_returns_empty_for_unknown_connection() -> None:
    store = _make_store()
    assert store.events_for_call("no-such-conn") == []


def test_record_never_raises_when_db_path_is_invalid() -> None:
    store = CallEventStore(PersistenceSettings(db_path=Path("Z:\\nonexistent-drive\\call_events.db")))
    store.start_call("conn-1")  # must not raise - persistence is a side channel, never fatal
    store.record("conn-1", {"type": "transcript", "text": "hello"})
    store.end_call("conn-1")


def test_encrypted_events_round_trip_when_key_is_configured() -> None:
    key = Fernet.generate_key().decode("ascii")
    store = _make_store(encryption_key=key)
    store.start_call("conn-1")
    store.record("conn-1", {"type": "transcript", "text": "hello encrypted world"})
    store.end_call("conn-1")

    events = store.events_for_call("conn-1")

    assert events == [{"type": "transcript", "text": "hello encrypted world"}]


def test_encrypted_payload_is_not_stored_as_plaintext_in_the_db_file() -> None:
    key = Fernet.generate_key().decode("ascii")
    store = _make_store(encryption_key=key)
    secret_text = "patient MRN 12345 super secret plaintext marker"
    store.start_call("conn-1")
    store.record("conn-1", {"type": "transcript", "text": secret_text})

    # Bypass CallEventStore entirely and inspect the raw column value, so a
    # bug that just re-encodes (rather than actually encrypts) can't hide.
    connection = sqlite3.connect(store.settings.db_path)
    try:
        (raw_payload,) = connection.execute(
            "SELECT payload FROM call_events WHERE connection_id = ?", ("conn-1",)
        ).fetchone()
    finally:
        connection.close()

    assert secret_text not in raw_payload
    assert "transcript" not in raw_payload
    # Sanity check it's still readable through the real API.
    assert store.events_for_call("conn-1")[0]["text"] == secret_text


def test_invalid_encryption_key_raises_value_error_at_construction() -> None:
    with pytest.raises(ValueError):
        CallEventStore(
            PersistenceSettings(
                db_path=Path(tempfile.mkdtemp()) / "call_events.db",
                encryption_key="not-a-valid-fernet-key",
            )
        )


if __name__ == "__main__":
    test_load_is_idempotent()
    test_record_and_read_back_events_in_order()
    test_events_for_call_only_returns_that_connections_events()
    test_events_for_call_returns_empty_for_unknown_connection()
    test_record_never_raises_when_db_path_is_invalid()
    test_encrypted_events_round_trip_when_key_is_configured()
    test_encrypted_payload_is_not_stored_as_plaintext_in_the_db_file()
    test_invalid_encryption_key_raises_value_error_at_construction()
    print("ok")
