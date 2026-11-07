"""Flask API for the Leon chat.

    POST   /v1/messages                 Anthropic-compatible API (see anthropic_api.py): chat,
    POST   /v1/messages/count_tokens    raw completion and models, usable with the official
    POST   /v1/complete                 `anthropic` SDK by setting its base_url
    GET    /v1/models[/<id>]
    GET    /api/conversations           conversation history, most recent first
    POST   /api/conversations           create one: {"mode", "model", "title"}
    GET    /api/conversations/<id>      one conversation with its messages
    PATCH  /api/conversations/<id>      rename: {"title"}
    DELETE /api/conversations/<id>      delete it and its messages
    DELETE /api/conversations/<id>/messages?from=N
                                        drop messages N, N+1, ... (regenerate or edit the last turn)
    POST   /api/context                 how the next request fills the context (the web's meter):
                                        {"model", "mode", "system", "messages", "draft", "max_tokens"}
    GET    /                            the React app, once built (apps/web/dist)

Generation is saved in a conversation by sending the X-Leon-Conversation header to /v1.

Run: python apps/api/app.py     then open http://127.0.0.1:8000
"""

import argparse
from pathlib import Path

import torch
from anthropic_api import create_blueprint, error_response
from engine import MODES, Engine, find_checkpoints
from flask import Flask, abort, jsonify, request, send_from_directory
from store import ConversationStore
from werkzeug.security import safe_join

from leon.paths import DATA_DIR

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
DEFAULT_DB = DATA_DIR / "chat.db"


def bad_request(message: str):
    return jsonify({"error": message}), 400


def create_app(engine: Engine, store: ConversationStore, port: int) -> Flask:
    app = Flask(__name__, static_folder=None)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @app.before_request
    def only_local_hosts():
        # Binding to 127.0.0.1 is not enough on its own: a malicious web page could point its own
        # domain at 127.0.0.1 ("DNS rebinding") and then call this API from your browser.
        # Such requests carry that domain in the Host header, so they are rejected here.
        if request.host not in allowed_hosts:
            if request.path.startswith("/v1/"):
                return error_response(f"host not allowed: {request.host}", 403)
            abort(403)

    app.register_blueprint(create_blueprint(engine, store))

    # --- conversation history ---------------------------------------------------------

    @app.get("/api/conversations")
    def list_conversations():
        return jsonify({"conversations": store.list()})

    @app.post("/api/conversations")
    def create_conversation():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or body.get("mode") not in MODES or not isinstance(body.get("model"), str):
            return bad_request(f"needs 'mode' ({' or '.join(MODES)}) and 'model'")
        title = body.get("title", "")
        if not isinstance(title, str):
            return bad_request("'title' must be a string")
        return jsonify(store.create(body["mode"], body["model"], title)), 201

    @app.get("/api/conversations/<conversation_id>")
    def get_conversation(conversation_id: str):
        conversation = store.get(conversation_id)
        return jsonify(conversation) if conversation else (jsonify({"error": "not found"}), 404)

    @app.patch("/api/conversations/<conversation_id>")
    def rename_conversation(conversation_id: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("title"), str) or not body["title"].strip():
            return bad_request("needs a non-empty string 'title'")
        if not store.rename(conversation_id, body["title"]):
            return jsonify({"error": "not found"}), 404
        return jsonify(store.get(conversation_id))

    @app.delete("/api/conversations/<conversation_id>")
    def delete_conversation(conversation_id: str):
        return ("", 204) if store.delete(conversation_id) else (jsonify({"error": "not found"}), 404)

    @app.delete("/api/conversations/<conversation_id>/messages")
    def truncate_conversation(conversation_id: str):
        """Drop the messages from index ?from=N on: regenerating or editing the last exchange
        replaces it, instead of leaving the old attempt in the history."""
        keep = request.args.get("from", type=int)
        if keep is None or keep < 0:
            return bad_request("needs a non-negative integer query parameter 'from'")
        if not store.truncate(conversation_id, keep):
            return jsonify({"error": "not found"}), 404
        return jsonify(store.get(conversation_id))

    @app.post("/api/context")
    def context_usage():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or body.get("model") not in find_checkpoints():
            return bad_request("needs the 'model' of an existing checkpoint")
        if body.get("mode", "chat") not in MODES:
            return bad_request(f"'mode' must be {' or '.join(MODES)}")
        system, draft, messages = body.get("system", ""), body.get("draft", ""), body.get("messages", [])
        if not isinstance(system, str) or not isinstance(draft, str):
            return bad_request("'system' and 'draft' must be strings")
        valid = isinstance(messages, list) and all(
            isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
            for m in messages
        )
        if not valid:
            return bad_request("'messages' must be a list of {role: user | assistant, content: string}")
        if not isinstance(body.get("tools", False), bool):
            return bad_request("'tools' must be true or false")
        max_tokens = body.get("max_tokens", 256)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            return bad_request("'max_tokens' must be a positive integer")
        usage = engine.context_usage(
            body["model"],
            body.get("mode", "chat"),
            system.strip(),
            messages,
            draft.strip(),
            max_tokens,
            tools=body.get("tools", False),
        )
        return jsonify(usage)

    @app.route("/api/<path:_>", methods=["GET", "POST", "PATCH", "DELETE"])
    def unknown_api(_):
        return jsonify({"error": "not found"}), 404

    # --- web app ----------------------------------------------------------------------------

    @app.get("/")
    @app.get("/<path:path>")
    def web(path: str = "index.html"):
        if not (WEB_DIST / "index.html").exists():
            return (
                "<p>The web app is not built yet. Run <code>npm install</code> and <code>npm run build</code> "
                "in <code>apps/web</code>, or use <code>npm run dev</code> and open http://localhost:5173.</p>",
                503,
            )
        file = safe_join(str(WEB_DIST), path)  # None if the path tries to leave WEB_DIST
        if file is not None and Path(file).is_file():
            return send_from_directory(WEB_DIST, path)
        return send_from_directory(WEB_DIST, "index.html")  # single-page app: the client routes

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite file with the conversation history")
    args = parser.parse_args()

    if not find_checkpoints():
        print("No Leon checkpoints yet. Train one first: python phases/3_modern/train.py")
    app = create_app(Engine(args.device), ConversationStore(args.db), args.port)
    print(f"Leon chat API on http://127.0.0.1:{args.port}  (device: {args.device}, history: {args.db})")
    # Development server, single machine: 127.0.0.1 is reachable only from this computer.
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
