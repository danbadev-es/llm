"""The Anthropic-compatible /v1 API, driven by the official `anthropic` SDK without a network:
the SDK's HTTP client calls the Flask app in-process (WSGI). Same tiny checkpoint as test_api.py.
Run: pytest tests/test_anthropic_api.py"""

import tempfile
from pathlib import Path

import anthropic
import httpx2
import pytest
import torch
from test_api import HOST, MODEL, PORT, engine, make_app  # test_api puts apps/api on sys.path

from leon.chat import encode_chat

GREEDY = {"temperature": 0}  # SDK 2.x removed the sampling parameters: they go in extra_body
HELLO = [{"role": "user", "content": "hola"}]


@pytest.fixture(scope="module")
def app():
    original_dir, original_loader = engine.CHECKPOINT_DIR, engine.load_tokenizer
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        yield make_app(Path(tmp))
    engine.CHECKPOINT_DIR, engine.load_tokenizer = original_dir, original_loader


@pytest.fixture(scope="module")
def client(app):
    http = httpx2.Client(transport=httpx2.WSGITransport(app=app))
    return anthropic.Anthropic(api_key="local", base_url=f"http://127.0.0.1:{PORT}", http_client=http, max_retries=0)


def test_messages_create(client) -> None:
    torch.manual_seed(0)
    message = client.messages.create(model=MODEL, max_tokens=16, system="Sé breve.", messages=HELLO)
    assert message.id.startswith("msg_") and message.type == "message" and message.role == "assistant"
    assert message.model == MODEL and message.content[0].type == "text"
    assert message.stop_reason in ("end_turn", "max_tokens") and message.stop_sequence is None
    assert (
        message.usage.input_tokens
        == client.messages.count_tokens(model=MODEL, system="Sé breve.", messages=HELLO).input_tokens
    )
    assert message.leon["tokens"] == message.usage.output_tokens  # Leon's extra stats


def test_messages_stream(client) -> None:
    torch.manual_seed(0)
    with client.messages.stream(model=MODEL, max_tokens=16, messages=HELLO) as stream:
        events = list(stream)
    order = [e.type for e in events if e.type not in ("text", "content_block_delta")]  # the SDK hides ping
    assert order == ["message_start", "content_block_start", "content_block_stop", "message_delta", "message_stop"]
    assert any(e.type == "content_block_delta" and e.delta.type == "text_delta" for e in events)
    final = stream.get_final_message()
    delta = next(e for e in events if e.type == "message_delta")
    assert delta.leon["tokens"] == final.usage.output_tokens and "tok_per_s" in delta.leon

    torch.manual_seed(0)  # same seed: the same sampled answer, now read as text_stream
    with client.messages.stream(model=MODEL, max_tokens=16, messages=HELLO) as stream:
        assert "".join(stream.text_stream) == final.content[0].text
    torch.manual_seed(0)
    assert client.messages.create(model=MODEL, max_tokens=16, messages=HELLO).content[0].text == final.content[0].text


def test_raw_stream_event_order(app) -> None:
    body = {"model": MODEL, "max_tokens": 4, "messages": HELLO, "stream": True, **GREEDY}
    text = app.test_client().post("/v1/messages", json=body, headers=HOST).get_data(as_text=True)
    names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    names = [n for i, n in enumerate(names) if n != "content_block_delta" or names[i - 1] != n]  # one delta
    start = ["message_start", "content_block_start", "ping"]
    end = ["content_block_stop", "message_delta", "message_stop"]
    assert names in (start + end, start + ["content_block_delta"] + end)


def test_stop_sequence_helpers() -> None:
    assert engine._find_stop("abcdcd", ["cd", "x"]) == (2, "cd")
    assert engine._find_stop("abc", ["x"]) == (3, None)
    assert engine._partial_stop("hola ad", ["adiós"]) == 2  # "ad" may still become "adiós": hold it back
    assert engine._partial_stop("hola", ["adiós"]) == 1  # even a final "a" could be its start
    assert engine._partial_stop("hole", ["adiós"]) == 0


def test_stop_sequences(client) -> None:
    torch.manual_seed(0)
    full = client.messages.create(model=MODEL, max_tokens=16, messages=HELLO).content[0].text
    assert len(full) >= 4, "the tiny model should write a few characters with this seed"
    stop = full[1:3]
    for streaming in (False, True):
        torch.manual_seed(0)
        if streaming:
            with client.messages.stream(model=MODEL, max_tokens=16, messages=HELLO, stop_sequences=[stop]) as s:
                text = "".join(s.text_stream)
                message = s.get_final_message()
        else:
            message = client.messages.create(model=MODEL, max_tokens=16, messages=HELLO, stop_sequences=[stop])
            text = message.content[0].text
        assert text == full[: full.find(stop)]  # the sequence itself is not part of the answer
        assert message.stop_reason == "stop_sequence" and message.stop_sequence == stop

    # Complete mode, greedy: the tiny model repeats "hola", so "laho" stops it after "ho".
    body = {"model": MODEL, "prompt": "hola", "max_tokens_to_sample": 8, "stop_sequences": ["laho"], **GREEDY}
    result = client.post("/v1/complete", cast_to=object, body=body)
    assert (result["completion"], result["stop_reason"], result["stop"]) == ("ho", "stop_sequence", "laho")


def test_prefill_continues_the_assistant(client) -> None:
    messages = HELLO + [{"role": "assistant", "content": "qué"}]
    message = client.messages.create(model=MODEL, max_tokens=4, messages=messages, extra_body=GREEDY)
    expected = encode_chat(engine.load_tokenizer(), messages, continue_final_message=True)
    assert message.usage.input_tokens == len(expected)  # "<|user|>hola<|end|><|assistant|>qué", left open
    assert client.messages.count_tokens(model=MODEL, messages=messages).input_tokens == len(expected)
    with pytest.raises(anthropic.BadRequestError):
        client.messages.create(model=MODEL, max_tokens=4, messages=HELLO + [{"role": "assistant", "content": "qué "}])


def test_legacy_complete_stream(client) -> None:
    body = {"model": MODEL, "prompt": "hola", "max_tokens_to_sample": 8, "stream": True, **GREEDY}
    # SDK 2.x dropped client.completions: the generic post() reads the same stream
    events = list(
        client.post("/v1/complete", cast_to=object, body=body, stream=True, stream_cls=anthropic.Stream[object])
    )
    assert all(e["type"] == "completion" and e["id"].startswith("compl_") for e in events)
    assert all(e["stop_reason"] is None for e in events[:-1])
    assert "".join(e["completion"] for e in events) == "hola" * 8
    assert events[-1]["stop_reason"] == "max_tokens" and events[-1]["leon"]["tokens"] == 8
    plain = client.post("/v1/complete", cast_to=object, body={**body, "stream": False})
    assert plain["completion"] == "hola" * 8 and plain["type"] == "completion"


def test_models(client) -> None:
    models = list(client.models.list())
    assert [m.id for m in models] == [MODEL]
    model = client.models.retrieve(MODEL)
    assert model.type == "model" and model.max_input_tokens == 32 and model.max_tokens == 16
    assert model.leon == {
        "params": model.leon["params"],
        "context": 32,
        "max_output_tokens": 16,
        "step": 1,
        "stage": "base",
    }
    with pytest.raises(anthropic.NotFoundError):
        client.models.retrieve("3_test/nope")


def test_errors(client, app, monkeypatch) -> None:
    with pytest.raises(anthropic.NotFoundError) as error:
        client.messages.create(model="nope", max_tokens=4, messages=HELLO)
    assert error.value.body["error"]["type"] == "not_found_error"
    bad_requests = (
        {"messages": HELLO, "extra_body": {"temperature": 9}},
        {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
        {"messages": [{"role": "system", "content": "x"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": HELLO, "stop_sequences": [""]},
    )
    for kwargs in bad_requests:
        with pytest.raises(anthropic.BadRequestError) as error:
            client.messages.create(model=MODEL, max_tokens=4, **kwargs)
        assert error.value.body["error"]["type"] == "invalid_request_error", kwargs

    # The raw envelope, for what the SDK cannot send (no max_tokens) or other hosts.
    raw = app.test_client()
    response = raw.post("/v1/messages", json={"model": MODEL, "messages": HELLO}, headers=HOST)
    assert response.status_code == 400
    assert response.get_json() == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "max_tokens: Field required"},
    }
    assert (
        raw.get("/v1/models", headers={"Host": "evil.example:8000"}).get_json()["error"]["type"] == "permission_error"
    )
    assert raw.get("/v1/nope", headers=HOST).get_json()["error"]["type"] == "not_found_error"

    # A failure while generating: 500 api_error, or an `error` event once the stream has started.
    def broken(*args, **kwargs):
        raise RuntimeError("boom")
        yield

    monkeypatch.setattr(engine.Leon, "stream", broken)
    with pytest.raises(anthropic.InternalServerError):
        client.messages.create(model=MODEL, max_tokens=4, messages=HELLO)
    with pytest.raises(anthropic.APIStatusError, match="boom"):
        with client.messages.stream(model=MODEL, max_tokens=4, messages=HELLO) as stream:
            list(stream)


def test_conversation_header_saves_the_exchange(client, app) -> None:
    raw = app.test_client()
    chat = raw.post("/api/conversations", json={"mode": "chat", "model": MODEL}, headers=HOST).get_json()
    header = {"X-Leon-Conversation": chat["id"]}
    torch.manual_seed(0)
    message = client.messages.create(model=MODEL, max_tokens=8, messages=HELLO, extra_headers=header)
    saved = raw.get(f"/api/conversations/{chat['id']}", headers=HOST).get_json()["messages"]
    assert [m["role"] for m in saved] == ["user", "assistant"] and saved[1]["content"] == message.content[0].text
    assert saved[1]["stats"]["tokens"] == message.usage.output_tokens and "stopped" in saved[1]["stats"]

    # Hanging up mid-stream saves the partial answer as stopped by the user.
    completion = raw.post("/api/conversations", json={"mode": "complete", "model": MODEL}, headers=HOST).get_json()
    body = {"model": MODEL, "prompt": "hola", "max_tokens_to_sample": 16, "stream": True, **GREEDY}
    options = {"headers": {"X-Leon-Conversation": completion["id"]}}
    stream = client.post(
        "/v1/complete", cast_to=object, body=body, options=options, stream=True, stream_cls=anthropic.Stream[object]
    )
    first = next(iter(stream))
    stream.close()
    saved = raw.get(f"/api/conversations/{completion['id']}", headers=HOST).get_json()["messages"]
    assert len(saved) == 1 and saved[0]["stats"] == {"stopped": "user"} and saved[0]["content"].startswith("hola")
    assert first["type"] == "completion"

    # Unknown conversation, or one of the other mode.
    with pytest.raises(anthropic.NotFoundError):
        client.messages.create(model=MODEL, max_tokens=4, messages=HELLO, extra_headers={"X-Leon-Conversation": "x"})
    with pytest.raises(anthropic.BadRequestError):
        client.post("/v1/complete", cast_to=object, body={**body, "stream": False}, options={"headers": header})
