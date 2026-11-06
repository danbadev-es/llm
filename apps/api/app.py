"""Flask API for the Leon chat.

    GET    /api/models                  available Leon checkpoints (phase 3 onwards)
    POST   /api/generate                streams the generated text as Server-Sent Events
    GET    /api/conversations           conversation history, most recent first
    POST   /api/conversations           create one: {"mode", "model", "title"}
    GET    /api/conversations/<id>      one conversation with its messages
    PATCH  /api/conversations/<id>      rename: {"title"}
    DELETE /api/conversations/<id>      delete it and its messages
    GET    /                            the React app, once built (apps/web/dist)

Run: python apps/api/app.py     then open http://127.0.0.1:8000
"""

import argparse
import json
from pathlib import Path

import torch
from engine import MODES, Engine, find_checkpoints, validate
from flask import Flask, Response, abort, jsonify, request, send_from_directory, stream_with_context
from store import ConversationStore
from werkzeug.security import safe_join

from leon.paths import DATA_DIR

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
DEFAULT_DB = DATA_DIR / "chat.db"


def sse(event: dict) -> str:
    """One Server-Sent Event: a "data: <json>" line followed by a blank line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


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
            abort(403)

    @app.get("/api/models")
    def models():
        return jsonify({"models": engine.list_models(), "device": engine.device})

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

    # --- generation -----------------------------------------------------------------------

    @app.post("/api/generate")
    def generate():
        body = request.get_json(silent=True)
        try:
            params = validate(body)
        except ValueError as error:
            return bad_request(str(error))

        # Optional: save this exchange in a conversation of the history.
        conversation_id = body.get("conversation_id")
        if conversation_id is not None:
            conversation = store.get(conversation_id) if isinstance(conversation_id, str) else None
            if conversation is None:
                return bad_request("unknown 'conversation_id'")
            if conversation["mode"] != params["mode"]:
                return bad_request(f"this conversation is in '{conversation['mode']}' mode")
            if params["mode"] == "chat" and params["messages"][-1]["role"] != "user":
                return bad_request("in chat mode the last message must be the user's")

        def events():
            text, saved = [], False

            def save(**fields) -> None:
                nonlocal saved
                if conversation_id is None or saved:
                    return
                prompt = params["prompt"] if params["mode"] == "complete" else None
                store.add_message(
                    conversation_id, "assistant", "".join(text), prompt=prompt, model=params["model"], **fields
                )
                saved = True

            if conversation_id is not None and params["mode"] == "chat":
                store.add_message(conversation_id, "user", params["messages"][-1]["content"], model=params["model"])
            stream = engine.stream(params)
            try:
                for event in stream:
                    if "text" in event:
                        text.append(event["text"])
                    elif event.get("done"):
                        save(stats={k: v for k, v in event.items() if k != "done"})
                    yield sse(event)
            except Exception as error:  # headers are already sent: report it as an event
                save(error=f"{type(error).__name__}: {error}")
                yield sse({"error": f"{type(error).__name__}: {error}"})
            finally:
                # Runs when generation ends or when the browser disconnects ("stop" button):
                # closing the generator stops the model and releases the lock; a stopped answer
                # is saved with what had been written so far.
                stream.close()
                save(stats={"stopped": "user"})

        return Response(
            stream_with_context(events()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
