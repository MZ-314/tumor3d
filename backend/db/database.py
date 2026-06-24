"""SQLite persistence for chat sessions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config_medical import DB_PATH, ensure_data_dirs
from shared.schemas.pydantic.reconstruct import (
    ChatDetail,
    ChatMessageRecord,
    ChatSummary,
    ReconstructResponse,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    ensure_data_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT,
                attachment_url TEXT,
                reconstruction_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
            """
        )


def create_chat(title: str = "New scan") -> ChatSummary:
    chat_id = uuid.uuid4().hex[:12]
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, title, now, now),
        )
    return ChatSummary(id=chat_id, title=title, created_at=now, updated_at=now)


def list_chats(limit: int = 50) -> list[ChatSummary]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [ChatSummary(**dict(r)) for r in rows]


def get_chat(chat_id: str) -> ChatDetail | None:
    with _connect() as conn:
        chat = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
        if not chat:
            return None
        msg_rows = conn.execute(
            """
            SELECT id, role, text, attachment_url, reconstruction_json, created_at
            FROM messages WHERE chat_id = ? ORDER BY created_at ASC
            """,
            (chat_id,),
        ).fetchall()

    messages: list[ChatMessageRecord] = []
    for row in msg_rows:
        recon = None
        if row["reconstruction_json"]:
            recon = ReconstructResponse.model_validate_json(row["reconstruction_json"])
        messages.append(
            ChatMessageRecord(
                id=row["id"],
                role=row["role"],
                text=row["text"],
                attachment_url=row["attachment_url"],
                reconstruction=recon,
                created_at=row["created_at"],
            )
        )

    return ChatDetail(
        id=chat["id"],
        title=chat["title"],
        created_at=chat["created_at"],
        updated_at=chat["updated_at"],
        messages=messages,
    )


def touch_chat(chat_id: str, title: str | None = None) -> None:
    now = _now()
    with _connect() as conn:
        if title:
            conn.execute(
                "UPDATE chats SET updated_at = ?, title = ? WHERE id = ?",
                (now, title, chat_id),
            )
        else:
            conn.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?",
                (now, chat_id),
            )


def add_message(
    chat_id: str,
    role: str,
    *,
    text: str | None = None,
    attachment_url: str | None = None,
    reconstruction: ReconstructResponse | None = None,
) -> ChatMessageRecord:
    msg_id = uuid.uuid4().hex[:12]
    now = _now()
    recon_json = reconstruction.model_dump_json() if reconstruction else None
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, chat_id, role, text, attachment_url, reconstruction_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, chat_id, role, text, attachment_url, recon_json, now),
        )
    touch_chat(chat_id)
    return ChatMessageRecord(
        id=msg_id,
        role=role,
        text=text,
        attachment_url=attachment_url,
        reconstruction=reconstruction,
        created_at=now,
    )
