"""Checks for apps/api with Flask's test client: a tiny random checkpoint and a toy tokenizer,
so no data, trained model or GPU is needed. Generation itself is covered through the official
SDK in test_anthropic_api.py; this file covers the history and the server's safety rules.
Run: pytest tests/test_api.py"""

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


def make_app(tmp: Path):
    """The Flask app with an engine that sees a single tiny checkpoint (restore engine's globals after)."""
    tokenizer = BPETokenizer.train(count_chunks(["hola qué tal, muy bien gracias"] * 5), 300)
    config = LeonConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_layer=1, n_head=2, n_kv_head=1, n_embd=16)
    torch.manual_seed(0)
    save_checkpoint(tmp / "3_test" / "tiny" / "best.pt", Leon(config), step=1, best_val=0.0)
    engine.CHECKPOINT_DIR = tmp
    engine.load_tokenizer = lambda *_: tokenizer  # every tokenizer version resolves to the toy one
    store = ConversationStore(tmp / "chat.db")
    return api_app.create_app(engine.Engine("cpu"), store, PORT)


def with_client(test) -> None:
    original_dir, original_loader = engine.CHECKPOINT_DIR, engine.load_tokenizer
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        try:
            test(make_app(Path(tmp)).test_client())
        finally:
            engine.CHECKPOINT_DIR, engine.load_tokenizer = original_dir, original_loader


def test_safety_rules() -> None:
    def check(client) -> None:
        # Other hosts (DNS rebinding) are refused, on /v1 with Anthropic's error envelope.
        assert client.get("/api/conversations", headers={"Host": "evil.example:8000"}).status_code == 403
        refused = client.get("/v1/models", headers={"Host": "evil.example:8000"})
        assert refused.status_code == 403 and refused.get_json()["error"]["type"] == "permission_error"
        # Unknown API paths are 404, never the web page; the pre-/v1 endpoints are gone.
        for path in ("/api/nope", "/api/generate", "/api/models"):
            assert client.get(path, headers=HOST).status_code == 404, path
        assert client.post("/api/generate", json={}, headers=HOST).status_code == 404
        # Paths cannot escape the web build folder (no file-existence oracle either).
        assert client.get("/..%2f..%2fapi%2fapp.py", headers=HOST).status_code in (200, 503)

    with_client(check)


def test_conversation_history() -> None:
    def check(client) -> None:
        def post(url, body, **headers):
            return client.post(url, json=body, headers={**HOST, **headers})

        created = post("/api/conversations", {"mode": "chat", "model": MODEL, "title": "  hola\n qué   tal  "})
        assert created.status_code == 201
        chat = created.get_json()
        assert chat["title"] == "hola qué tal"  # whitespace collapsed into a one-line title
        long_title = post("/api/conversations", {"mode": "complete", "model": MODEL, "title": "x" * 200}).get_json()
        assert len(long_title["title"]) == 60 and long_title["title"].endswith("…")
        assert post("/api/conversations", {"mode": "x", "model": MODEL}).status_code == 400

        # Two chat exchanges saved through the X-Leon-Conversation header of /v1/messages.
        url = f"/api/conversations/{chat['id']}"
        torch.manual_seed(0)
        for question in ("hola", "qué tal"):
            history = client.get(url, headers=HOST).get_json()["messages"]
            messages = [{"role": m["role"], "content": m["content"] or "-"} for m in history]
            body = {"model": MODEL, "max_tokens": 4, "messages": [*messages, {"role": "user", "content": question}]}
            assert post("/v1/messages", body, **{"X-Leon-Conversation": chat["id"]}).status_code == 200
        saved = client.get(url, headers=HOST).get_json()["messages"]
        assert [m["role"] for m in saved] == ["user", "assistant", "user", "assistant"]
        assert saved[2]["content"] == "qué tal" and "tokens" in saved[3]["stats"]

        # The most recently used conversation comes first.
        listed = client.get("/api/conversations", headers=HOST).get_json()["conversations"]
        assert listed[0]["id"] == chat["id"]

        # Regenerating or editing the last exchange first drops it from the history.
        truncated = client.delete(f"{url}/messages?from=2", headers=HOST)
        assert [m["content"] for m in truncated.get_json()["messages"]] == ["hola", saved[1]["content"]]
        assert client.delete(f"{url}/messages", headers=HOST).status_code == 400
        assert client.delete("/api/conversations/nope/messages?from=0", headers=HOST).status_code == 404

        # Rename and delete.
        assert client.patch(url, json={"title": "Saludos"}, headers=HOST).get_json()["title"] == "Saludos"
        assert client.patch(url, json={"title": " "}, headers=HOST).status_code == 400
        assert client.delete(url, headers=HOST).status_code == 204
        assert client.get(url, headers=HOST).status_code == 404
        assert client.delete(url, headers=HOST).status_code == 404

    with_client(check)


if __name__ == "__main__":
    test_safety_rules()
    test_conversation_history()
    print("ok  test_safety_rules, test_conversation_history")
