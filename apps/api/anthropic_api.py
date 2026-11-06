"""Anthropic-compatible API: Leon behind the same HTTP interface as the Claude API.

Any Anthropic client can talk to Leon by changing its base URL, e.g. the Python SDK:

    client = anthropic.Anthropic(api_key="local", base_url="http://127.0.0.1:8000")
    client.messages.create(model="3_modern/wiki_32m", max_tokens=200, messages=[...])

    POST /v1/messages               chat mode (Messages API), streaming or not
    POST /v1/messages/count_tokens  {"input_tokens"} of a Messages request
    POST /v1/complete               complete mode (legacy Text Completions API): raw text continuation
    GET  /v1/models                 the checkpoints as Anthropic model objects
    GET  /v1/models/<id>            one of them

Differences from the real API, on purpose:
    - only text: image, document, tool and thinking blocks are a 400 (and there are no tools)
    - x-api-key, Authorization and anthropic-version are accepted and ignored (local server)
    - max_tokens above the model's limit (half its context) is clamped instead of rejected
    - temperature goes up to 5 and top_k may be 0 (no filter): a superset of the valid values
    - SDK 2.x no longer has temperature/top_k/top_p arguments nor client.completions:
      send them with extra_body={...} and call /v1/complete with client.post(...)

Leon extras, all under a "leon" key so that clients ignore them:
    - model objects: {"params", "context", "max_output_tokens", "step", "stage"}
    - the final stats of a generation (non-streaming message, final message_delta event, final
      completion event): {"tokens", "tok_per_s", "prompt_tokens", "dropped_turns", "stopped"}
    - request header X-Leon-Conversation: <id> saves the exchange in that conversation of the history
"""

import json
import uuid
from datetime import UTC, datetime

from engine import Engine, find_checkpoints, read_number
from flask import Blueprint, Response, jsonify, request
from store import ConversationStore
from werkzeug.exceptions import HTTPException

ERROR_TYPES = {400: "invalid_request_error", 403: "permission_error", 404: "not_found_error", 500: "api_error"}
STOP_REASONS = {"end": "end_turn", "max_tokens": "max_tokens", "stop_sequence": "stop_sequence"}
CONVERSATION_HEADER = "X-Leon-Conversation"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def error_body(message: str, status: int) -> dict:
    """Anthropic's error envelope, also used for the `error` event of a stream."""
    return {"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message}}


def error_response(message: str, status: int):
    return jsonify(error_body(message, status)), status


def sse(event: dict) -> str:
    """One Server-Sent Event in Anthropic's format: its type as the event name, then the JSON."""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


# --- requests: Anthropic body -> engine params ---------------------------------------------------


def _number(body: dict, key: str, default, low, high, integer: bool = False):
    if default is None and key not in body:
        raise ApiError(f"{key}: Field required")
    try:
        return read_number(body, key, default, low, high, integer)
    except ValueError as error:
        raise ApiError(str(error)) from None


def _text(content, where: str) -> str:
    """A string, or a list of text blocks ({"type": "text", "text": ...}) joined by blank lines."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str) for b in content
    ):
        return "\n\n".join(block["text"] for block in content)
    raise ApiError(f"{where}: must be a string or a list of text blocks (Leon only supports text)")


def _common(body) -> dict:
    """Fields shared by /v1/messages and /v1/complete: model, sampling and stop sequences."""
    if not isinstance(body, dict):
        raise ApiError("the body must be a JSON object")
    if not isinstance(body.get("model"), str):
        raise ApiError("model: Field required")
    if body["model"] not in find_checkpoints():
        raise ApiError(f"model: {body['model']}", 404)
    stops = body.get("stop_sequences", [])
    if not isinstance(stops, list) or not all(isinstance(s, str) and s for s in stops):
        raise ApiError("stop_sequences: must be a list of non-empty strings")
    top_p = body.get("top_p")
    return {
        "model": body["model"],
        "temperature": float(_number(body, "temperature", 1.0, 0.0, 5.0)),
        "top_k": _number(body, "top_k", 0, 0, 100_000, integer=True),  # 0 = no top-k filter
        "top_p": None if top_p is None else float(_number(body, "top_p", 1.0, 0.0, 1.0)),
        "stop_sequences": stops,
    }


def parse_messages(body, count_only: bool = False) -> dict:
    """A Messages API request as chat-mode params. count_only: no max_tokens (count_tokens)."""
    params = {**_common(body), "mode": "chat"}
    if not count_only:
        params["max_new_tokens"] = _number(body, "max_tokens", None, 1, 10**6, integer=True)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError("messages: must be a non-empty list")
    merged: list[dict] = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ApiError(f"messages.{i}.role: must be 'user' or 'assistant'")
        text = _text(message.get("content"), f"messages.{i}.content")
        if merged and merged[-1]["role"] == message["role"]:  # consecutive turns of one role are combined
            merged[-1]["content"] += "\n\n" + text
        else:
            merged.append({"role": message["role"], "content": text})
    for i, message in enumerate(merged[:-1]):
        if not message["content"]:
            raise ApiError(f"messages.{i}: all messages must have non-empty content except the final assistant one")
    final = merged[-1]
    if not final["content"] and final["role"] == "user":
        raise ApiError("messages: the final user message must have non-empty content")
    if final["role"] == "assistant" and final["content"] != final["content"].rstrip():
        raise ApiError("final assistant content cannot end with trailing whitespace")
    params["messages"] = merged
    params["prefill"] = final["role"] == "assistant"  # the answer continues the assistant's text
    params["system"] = _text(body.get("system", ""), "system").strip()
    return params


def parse_complete(body) -> dict:
    """A Text Completions request as complete-mode params. The prompt is raw text: no Human/Assistant markers."""
    params = {**_common(body), "mode": "complete"}
    params["max_new_tokens"] = _number(body, "max_tokens_to_sample", None, 1, 10**6, integer=True)
    if not isinstance(body.get("prompt"), str) or not body["prompt"]:
        raise ApiError("prompt: must be a non-empty string")
    params["prompt"] = body["prompt"]
    return params


# --- responses -------------------------------------------------------------------------------------


def leon_stats(done: dict) -> dict:
    return {k: done[k] for k in ("tokens", "tok_per_s", "prompt_tokens", "dropped_turns", "stopped")}


def model_object(info: dict) -> dict:
    saved = find_checkpoints()[info["name"]].stat().st_mtime  # when the checkpoint was last written
    return {
        "type": "model",
        "id": info["name"],
        "display_name": info["name"],
        "created_at": datetime.fromtimestamp(saved, UTC).isoformat().replace("+00:00", "Z"),
        "lifecycle": "active",
        "max_input_tokens": info["context"],
        "max_tokens": info["max_new_tokens"],
        "leon": {
            "params": info["params"],
            "context": info["context"],
            "max_output_tokens": info["max_new_tokens"],
            "step": info["step"],
            "stage": info["stage"],
        },
    }


def message_events(message_id: str, params: dict, events):
    """Engine events -> the Messages API stream: message_start, content_block_start, ping,
    content_block_delta..., content_block_stop, message_delta, message_stop."""
    for event in events:
        if event.get("started"):
            message = {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": params["model"],
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": event["prompt_tokens"], "output_tokens": 0},
            }
            yield {"type": "message_start", "message": message}
            yield {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
            yield {"type": "ping"}
        elif "text" in event:
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": event["text"]}}
        elif event.get("done"):
            yield {"type": "content_block_stop", "index": 0}
            delta = {"stop_reason": STOP_REASONS[event["stopped"]], "stop_sequence": event["stop_sequence"]}
            usage = {"input_tokens": event["prompt_tokens"], "output_tokens": event["tokens"]}
            yield {"type": "message_delta", "delta": delta, "usage": usage, "leon": leon_stats(event)}
            yield {"type": "message_stop"}


def new_message(message_id: str, model: str, text: str, done: dict) -> dict:
    """The Message object of a non-streaming response."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": STOP_REASONS[done["stopped"]],
        "stop_sequence": done["stop_sequence"],
        "usage": {"input_tokens": done["prompt_tokens"], "output_tokens": done["tokens"]},
        "leon": leon_stats(done),
    }


def completion(completion_id: str, model: str, text: str, done: dict | None) -> dict:
    """A legacy Completion object; the final one of a stream has the stop reason and the stats."""
    body = {"type": "completion", "id": completion_id, "completion": text, "model": model}
    if done is None:
        return {**body, "stop_reason": None, "stop": None}
    # Ending at <|endoftext|> counts as a stop sequence (like "\n\nHuman:" did), with stop = None.
    reason = "max_tokens" if done["stopped"] == "max_tokens" else "stop_sequence"
    return {**body, "stop_reason": reason, "stop": done["stop_sequence"], "leon": leon_stats(done)}


# --- routes ----------------------------------------------------------------------------------------


def create_blueprint(engine: Engine, store: ConversationStore) -> Blueprint:
    v1 = Blueprint("anthropic", __name__, url_prefix="/v1")

    @v1.errorhandler(ApiError)
    def api_error(error: ApiError):
        return error_response(str(error), error.status)

    @v1.errorhandler(Exception)
    def unexpected(error: Exception):
        if isinstance(error, HTTPException):
            return error_response(error.description, error.code)
        return error_response(f"{type(error).__name__}: {error}", 500)

    def generation(params: dict):
        """Engine events, saved in the history when the request names a conversation."""
        conversation_id = request.headers.get(CONVERSATION_HEADER)
        if conversation_id is None:
            return engine.stream(params)
        conversation = store.get(conversation_id)
        if conversation is None:
            raise ApiError(f"conversation not found: {conversation_id}", 404)
        if conversation["mode"] != params["mode"]:
            raise ApiError(f"this conversation is in '{conversation['mode']}' mode")
        if params.get("prefill"):
            raise ApiError(f"{CONVERSATION_HEADER} needs the user's message last (no prefill)")
        return store.record(conversation_id, params, engine.stream(params))

    def stream_response(chunks, events) -> Response:
        """SSE response. After the headers are sent, a failure becomes an `error` event."""

        def body():
            try:
                for chunk in chunks:
                    yield sse(chunk)
            except Exception as error:
                yield sse(error_body(f"{type(error).__name__}: {error}", 500))
            finally:
                events.close()  # also on disconnect: stops the model, saves a partial answer

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return Response(body(), mimetype="text/event-stream", headers=headers)

    def run(events) -> tuple[str, dict]:
        text, done = [], None
        for event in events:
            if "text" in event:
                text.append(event["text"])
            done = event if event.get("done") else done
        return "".join(text), done

    @v1.post("/messages")
    def messages():
        body = request.get_json(silent=True)
        params = parse_messages(body)
        events = generation(params)
        message_id = f"msg_{uuid.uuid4().hex}"
        if body.get("stream") is True:
            return stream_response(message_events(message_id, params, events), events)
        text, done = run(events)
        return jsonify(new_message(message_id, params["model"], text, done))

    @v1.post("/messages/count_tokens")
    def count_tokens():
        return jsonify({"input_tokens": engine.count_tokens(parse_messages(request.get_json(silent=True), True))})

    @v1.post("/complete")
    def complete():
        body = request.get_json(silent=True)
        params = parse_complete(body)
        events = generation(params)
        completion_id = f"compl_{uuid.uuid4().hex}"
        if body.get("stream") is True:

            def chunks():
                for event in events:
                    if event.get("started"):
                        yield {"type": "ping"}
                    elif "text" in event:
                        yield completion(completion_id, params["model"], event["text"], None)
                    elif event.get("done"):
                        yield completion(completion_id, params["model"], "", event)

            return stream_response(chunks(), events)
        text, done = run(events)
        return jsonify(completion(completion_id, params["model"], text, done))

    @v1.get("/models")
    def list_models():
        models = [model_object(info) for info in engine.list_models()]
        limit = request.args.get("limit", "20")
        if not limit.isdigit() or not 1 <= int(limit) <= 1000:
            raise ApiError("limit: must be an integer between 1 and 1000")
        ids = [m["id"] for m in models]
        after, before = request.args.get("after_id"), request.args.get("before_id")
        start = ids.index(after) + 1 if after in ids else 0
        end = ids.index(before) if before in ids else len(ids)
        window = models[start:end]
        page = window[-int(limit) :] if before else window[: int(limit)]
        return jsonify(
            {
                "data": page,
                "has_more": len(page) < len(window),
                "first_id": page[0]["id"] if page else None,
                "last_id": page[-1]["id"] if page else None,
            }
        )

    @v1.get("/models/<path:model_id>")
    def get_model(model_id: str):
        for info in engine.list_models():
            if info["name"] == model_id:
                return jsonify(model_object(info))
        raise ApiError(f"model: {model_id}", 404)

    @v1.route("/<path:_>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unknown(_):
        raise ApiError(f"not found: {request.path}", 404)

    return v1
