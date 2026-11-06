"""Conversation history in SQLite (standard library, a single file under data/).

Two tables:
    conversations   id, title, mode ("chat" | "complete"), model, created_at, updated_at
    messages        one row per bubble: role, content, and for answers the prompt they continued
                    (complete mode), their generation stats (JSON) and any error

Each call opens its own connection: Flask serves requests from several threads, and SQLite
connections must not be shared between threads.
"""

import json
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    mode        TEXT NOT NULL CHECK (mode IN ('chat', 'complete')),
    model       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    prompt           TEXT,
    stats            TEXT,
    error            TEXT,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, id);
"""
TITLE_LENGTH = 60


def title_from(text: str) -> str:
    """A default title: the first line of the first message, shortened."""
    line = " ".join(text.split())
    return line if len(line) <= TITLE_LENGTH else line[: TITLE_LENGTH - 1].rstrip() + "…"


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            db.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer

    @contextmanager
    def _connect(self):
        """A connection that commits on success, rolls back on error and is always closed."""
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")  # needed for ON DELETE CASCADE
            with connection:
                yield connection

    def list(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, title, mode, model, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def create(self, mode: str, model: str, title: str) -> dict:
        now = time.time()
        conversation = {
            "id": uuid.uuid4().hex,
            "title": title_from(title) or "Untitled",
            "mode": mode,
            "model": model,
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as db:
            db.execute(
                "INSERT INTO conversations VALUES (:id, :title, :mode, :model, :created_at, :updated_at)",
                conversation,
            )
        return conversation

    def get(self, conversation_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if row is None:
                return None
            messages = db.execute(
                "SELECT role, content, prompt, stats, error FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        result = []
        for message in messages:
            item = {k: v for k, v in dict(message).items() if v is not None}
            if "stats" in item:
                item["stats"] = json.loads(item["stats"])
            result.append(item)
        return {**dict(row), "messages": result}

    def rename(self, conversation_id: str, title: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title_from(title), conversation_id))
        return cursor.rowcount == 1

    def delete(self, conversation_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cursor.rowcount == 1

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        prompt: str | None = None,
        stats: dict | None = None,
        error: str | None = None,
        model: str | None = None,
    ) -> None:
        """Append a message and move the conversation to the top of the history."""
        now = time.time()
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages (conversation_id, role, content, prompt, stats, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, prompt, json.dumps(stats) if stats else None, error, now),
            )
            db.execute(
                "UPDATE conversations SET updated_at = ?, model = COALESCE(?, model) WHERE id = ?",
                (now, model, conversation_id),
            )
