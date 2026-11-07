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
from collections.abc import Iterator
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
    steps            TEXT,
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
            # Databases from before phase 6 lack the column of the tool calls: add it.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)")}
            if "steps" not in columns:
                db.execute("ALTER TABLE messages ADD COLUMN steps TEXT")

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
                "SELECT role, content, prompt, stats, error, steps FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        result = []
        for message in messages:
            item = {k: v for k, v in dict(message).items() if v is not None}
            for key in ("stats", "steps"):
                if key in item:
                    item[key] = json.loads(item[key])
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

    def truncate(self, conversation_id: str, keep: int) -> bool:
        """Keep only the first `keep` messages of a conversation (False if it does not exist)."""
        with self._connect() as db:
            if db.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is None:
                return False
            db.execute(
                "DELETE FROM messages WHERE conversation_id = ? AND id NOT IN "
                "(SELECT id FROM messages WHERE conversation_id = ? ORDER BY id LIMIT ?)",
                (conversation_id, conversation_id, keep),
            )
        return True

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        prompt: str | None = None,
        stats: dict | None = None,
        error: str | None = None,
        model: str | None = None,
        steps: list[dict] | None = None,
    ) -> None:
        """Append a message and move the conversation to the top of the history.

        steps: the tool calls of an answer, [{"id", "name", "input", "result", "at"}], where "at"
        is how much of the answer's text came before the call (where the web shows it).
        """
        now = time.time()
        encoded = [json.dumps(value, ensure_ascii=False) if value else None for value in (stats, steps)]
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages (conversation_id, role, content, prompt, stats, error, steps, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, prompt, encoded[0], error, encoded[1], now),
            )
            db.execute(
                "UPDATE conversations SET updated_at = ?, model = COALESCE(?, model) WHERE id = ?",
                (now, model, conversation_id),
            )

    def record(self, conversation_id: str, params: dict, stream: Iterator[dict]) -> Iterator[dict]:
        """Pass the events of Engine.stream() through, saving the exchange in a conversation.

        In chat mode the last message (the user's) is saved first. The answer is saved once:
        with the final stats, with the error if generation fails (the error is re-raised for
        the endpoint to report), or with what was written so far and {"stopped": "user"} if the
        client disconnects (closing this generator also closes the engine's, freeing the model).
        """
        text, steps, saved = [], [], False

        def save(**fields) -> None:
            nonlocal saved
            if not saved:
                prompt = params["prompt"] if params["mode"] == "complete" else None
                self.add_message(
                    conversation_id,
                    "assistant",
                    "".join(text),
                    prompt=prompt,
                    model=params["model"],
                    steps=steps,
                    **fields,
                )
                saved = True

        if params["mode"] == "chat":
            self.add_message(conversation_id, "user", params["messages"][-1]["content"], model=params["model"])
        try:
            for event in stream:
                if "text" in event:
                    text.append(event["text"])
                elif "tool_call" in event:
                    call = event["tool_call"]
                    steps.append({"id": event["id"], "name": call.get("name", ""),
                                  "input": call.get("arguments", {"raw": call.get("raw")}),
                                  "at": len("".join(text))})  # fmt: skip
                elif "tool_result" in event and steps:
                    steps[-1]["result"] = event["tool_result"]
                elif event.get("done"):
                    save(stats={k: v for k, v in event.items() if k != "done"})
                yield event
        except Exception as error:
            save(error=f"{type(error).__name__}: {error}")
            raise
        finally:
            stream.close()
            save(stats={"stopped": "user"})
