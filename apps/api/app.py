"""Flask API for the Leon chat.

    GET  /api/models     available Leon checkpoints (phase 3 onwards)
    POST /api/generate   streams the generated text as Server-Sent Events
    GET  /               the React app, once built (apps/web/dist)

Run: python apps/api/app.py     then open http://127.0.0.1:8000
"""

import argparse
import json
from pathlib import Path

import torch
from engine import Engine, find_checkpoints, validate
from flask import Flask, Response, abort, jsonify, request, send_from_directory, stream_with_context
from werkzeug.security import safe_join

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


def sse(event: dict) -> str:
    """One Server-Sent Event: a "data: <json>" line followed by a blank line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(engine: Engine, port: int) -> Flask:
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

    @app.post("/api/generate")
    def generate():
        try:
            params = validate(request.get_json(silent=True))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        def events():
            stream = engine.stream(params)
            try:
                for event in stream:
                    yield sse(event)
            except Exception as error:  # headers are already sent: report it as an event
                yield sse({"error": f"{type(error).__name__}: {error}"})
            finally:
                # Runs when generation ends or when the browser disconnects ("stop" button):
                # closing the generator stops the model and releases the lock.
                stream.close()

        return Response(
            stream_with_context(events()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/<path:_>", methods=["GET", "POST"])
    def unknown_api(_):
        return jsonify({"error": "not found"}), 404

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
    args = parser.parse_args()

    if not find_checkpoints():
        print("No Leon checkpoints yet. Train one first: python phases/3_modern/train.py")
    app = create_app(Engine(args.device), args.port)
    print(f"Leon chat API on http://127.0.0.1:{args.port}  (device: {args.device})")
    # Development server, single machine: 127.0.0.1 is reachable only from this computer.
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
