"""Checks for apps/api with Flask's test client: a tiny random checkpoint and a toy tokenizer,
so no data, trained model or GPU is needed. Run: pytest tests/test_api.py"""

import json
import sys
import tempfile
from pathlib import Path

import torch

from leon.model import Leon, LeonConfig
from leon.tokenizer import BPETokenizer, count_chunks
from leon.training import save_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))  # apps are not a package
import app as api_app  # noqa: E402
import engine  # noqa: E402
from store import ConversationStore  # noqa: E402

PORT = 8000
HOST = {"Host": f"127.0.0.1:{PORT}"}
MODEL = "3_test/tiny"


def make_client(tmp: Path):
    """A Flask test client whose engine sees a single tiny checkpoint."""
    tokenizer = BPETokenizer.train(count_chunks(["hola qué tal, muy bien gracias"] * 5), 300)
    config = LeonConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_layer=1, n_head=2, n_kv_head=1, n_embd=16)
    torch.manual_seed(0)
    save_checkpoint(tmp / "3_test" / "tiny" / "best.pt", Leon(config), step=1, best_val=0.0)
    engine.CHECKPOINT_DIR = tmp
    engine.load_tokenizer = lambda: tokenizer
    store = ConversationStore(tmp / "chat.db")
    return api_app.create_app(engine.Engine("cpu"), store, PORT).test_client()


def events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.get_data(as_text=True).split("\n\n") if line.startswith("data: ")]


def test_api() -> None:
    original_dir, original_loader = engine.CHECKPOINT_DIR, engine.load_tokenizer
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        try:
            client = make_client(Path(tmp))

            models = client.get("/api/models", headers=HOST).get_json()["models"]
            assert [m["name"] for m in models] == [MODEL] and models[0]["max_new_tokens"] == 16

            # Streaming generation in both modes ends with a "done" event.
            for body in (
                {"model": MODEL, "mode": "complete", "prompt": "hola", "max_new_tokens": 5},
                {"model": MODEL, "mode": "chat", "messages": [{"role": "user", "content": "hola"}], "system": "x"},
            ):
                response = client.post("/api/generate", json=body, headers=HOST)
                assert response.status_code == 200 and response.mimetype == "text/event-stream"
                assert events(response)[-1]["done"] is True

            # Long conversations drop whole old turns instead of cutting the context mid-message.
            history = [{"role": r, "content": "hola qué tal muy bien gracias " * 2} for r in ("user", "assistant") * 4]
            done = events(
                client.post(
                    "/api/generate",
                    json={"model": MODEL, "mode": "chat", "messages": history + [{"role": "user", "content": "hola"}]},
                    headers=HOST,
                )
            )[-1]
            assert done["dropped_turns"] > 0 and done["prompt_tokens"] <= 32 - 16

            # Every malformed request is a 400 before streaming starts.
            for bad in (
                [1],
                {"model": "nope", "mode": "complete", "prompt": "a"},
                {"model": MODEL, "mode": "x", "prompt": "a"},
                {"model": MODEL, "mode": "complete", "prompt": ""},
                {"model": MODEL, "mode": "complete", "prompt": "a", "top_k": -5},
                {"model": MODEL, "mode": "complete", "prompt": "a", "temperature": "abc"},
                {"model": MODEL, "mode": "complete", "prompt": "a", "temperature": -1},
                {"model": MODEL, "mode": "complete", "prompt": "a", "max_new_tokens": 2.5},
                {"model": MODEL, "mode": "chat", "messages": ["x"]},
                {"model": MODEL, "mode": "chat", "messages": [{"role": "system", "content": "x"}]},
                {"model": MODEL, "mode": "chat", "messages": [{"role": "user", "content": "x"}], "system": 5},
            ):
                response = client.post("/api/generate", json=bad, headers=HOST)
                assert response.status_code == 400 and "error" in response.get_json(), bad

            # Other hosts (DNS rebinding) are refused; unknown API paths are 404, not the web page.
            assert client.get("/api/models", headers={"Host": "evil.example:8000"}).status_code == 403
            assert client.get("/api/nope", headers=HOST).status_code == 404
            assert client.get("/api/generate", headers=HOST).status_code == 404
            # Paths cannot escape the web build folder (no file-existence oracle either).
            assert client.get("/..%2f..%2fapi%2fapp.py", headers=HOST).status_code in (200, 503)
        finally:
            engine.CHECKPOINT_DIR, engine.load_tokenizer = original_dir, original_loader


def test_conversation_history() -> None:
    original_dir, original_loader = engine.CHECKPOINT_DIR, engine.load_tokenizer
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        try:
            client = make_client(Path(tmp))
            post = lambda url, body: client.post(url, json=body, headers=HOST)  # noqa: E731

            created = post("/api/conversations", {"mode": "chat", "model": MODEL, "title": "  hola\n qué   tal  "})
            assert created.status_code == 201
            chat = created.get_json()
            assert chat["title"] == "hola qué tal"  # whitespace collapsed into a one-line title
            long_title = {"mode": "complete", "model": MODEL, "title": "x" * 200}
            completion = post("/api/conversations", long_title).get_json()
            assert len(completion["title"]) == 60 and completion["title"].endswith("…")

            # A chat exchange saves the user's turn and the answer with its stats.
            body = {"model": MODEL, "mode": "chat", "messages": [{"role": "user", "content": "hola"}]}
            done = events(post("/api/generate", {**body, "conversation_id": chat["id"], "max_new_tokens": 4}))[-1]
            saved = client.get(f"/api/conversations/{chat['id']}", headers=HOST).get_json()["messages"]
            assert [m["role"] for m in saved] == ["user", "assistant"] and saved[0]["content"] == "hola"
            assert saved[1]["stats"]["tokens"] == done["tokens"]

            # A completion saves one answer that remembers the prompt it continued.
            body = {"model": MODEL, "mode": "complete", "prompt": "hola", "max_new_tokens": 3}
            events(post("/api/generate", {**body, "conversation_id": completion["id"]}))
            saved = client.get(f"/api/conversations/{completion['id']}", headers=HOST).get_json()["messages"]
            assert len(saved) == 1 and saved[0]["prompt"] == "hola"

            # Stopping mid-answer (the browser closes the connection) still saves what was written.
            stop_me = post("/api/conversations", {"mode": "complete", "model": MODEL}).get_json()
            response = client.post(
                "/api/generate",
                json={**body, "max_new_tokens": 16, "conversation_id": stop_me["id"]},
                headers=HOST,
                buffered=False,
            )
            next(iter(response.response))  # read the first event, then hang up
            response.close()
            saved = client.get(f"/api/conversations/{stop_me['id']}", headers=HOST).get_json()["messages"]
            assert len(saved) == 1 and saved[0]["stats"] == {"stopped": "user"}
            client.delete(f"/api/conversations/{stop_me['id']}", headers=HOST)

            # The most recently used conversation comes first.
            listed = client.get("/api/conversations", headers=HOST).get_json()["conversations"]
            assert [c["id"] for c in listed] == [completion["id"], chat["id"]]

            # Saving into a missing conversation, or one of the other mode, is refused up front.
            assert post("/api/generate", {**body, "conversation_id": "nope"}).status_code == 400
            assert post("/api/generate", {**body, "conversation_id": chat["id"]}).status_code == 400

            # Rename and delete.
            url = f"/api/conversations/{chat['id']}"
            assert client.patch(url, json={"title": "Saludos"}, headers=HOST).get_json()["title"] == "Saludos"
            assert client.patch(url, json={"title": " "}, headers=HOST).status_code == 400
            assert client.delete(url, headers=HOST).status_code == 204
            assert client.get(url, headers=HOST).status_code == 404
            assert client.delete(url, headers=HOST).status_code == 404
            assert post("/api/conversations", {"mode": "x", "model": MODEL}).status_code == 400
        finally:
            engine.CHECKPOINT_DIR, engine.load_tokenizer = original_dir, original_loader


if __name__ == "__main__":
    test_api()
    test_conversation_history()
    print("ok  test_api, test_conversation_history")
