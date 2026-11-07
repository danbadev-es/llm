"""Tools in the /v1 API (phase 6), with the official `anthropic` SDK and a scripted model.

The model is replaced by a script of token ids (a random tiny model would never write a valid
call): what is tested is the API around it. Client tools end the answer with a tool_use block,
Leon's own tools run on the server, and the history keeps the calls.
Run: pytest tests/test_api_tools.py"""

import json
import tempfile
from pathlib import Path

import anthropic
import httpx2
import pytest
import torch
from test_api import HOST, PORT, ConversationStore, api_app, engine  # test_api puts apps/api on sys.path

from leon.model import Leon, LeonConfig
from leon.tokenizer import BPETokenizer, count_chunks
from leon.training import save_checkpoint

MODEL = "6_test/agent"
TEXT = 'hola {"name": "calculadora", "arguments": {"expresion": "17*24"}} son 408 tiempo Lima'
TOKENIZER = BPETokenizer.train(count_chunks([TEXT] * 5), 400)
S = TOKENIZER.special_tokens
WEATHER = {
    "name": "tiempo",
    "description": "el tiempo que hace",
    "input_schema": {"type": "object", "properties": {"ciudad": {"type": "string", "description": "dónde"}}},
}


@pytest.fixture()
def app():
    original_dir, original_loader = engine.CHECKPOINT_DIR, engine.load_tokenizer
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        config = LeonConfig(
            vocab_size=TOKENIZER.vocab_size, block_size=512, n_layer=1, n_head=2, n_kv_head=1, n_embd=16
        )
        save_checkpoint(tmp / "6_test" / "agent" / "best.pt", Leon(config), step=1, best_val=0.0, stage="agent")
        engine.CHECKPOINT_DIR = tmp
        engine.load_tokenizer = lambda *_: TOKENIZER
        yield api_app.create_app(engine.Engine("cpu"), ConversationStore(tmp / "chat.db"), PORT)
    engine.CHECKPOINT_DIR, engine.load_tokenizer = original_dir, original_loader


@pytest.fixture()
def client(app):
    http = httpx2.Client(transport=httpx2.WSGITransport(app=app))
    return anthropic.Anthropic(api_key="local", base_url=f"http://127.0.0.1:{PORT}", http_client=http, max_retries=0)


def script(monkeypatch, *segments: list[int]) -> list[list[int]]:
    """Replace the model: each generation replays the next segment; returns the prompts it saw."""
    remaining, prompts = iter(segments), []

    def stream(self, idx, max_new_tokens, stop_ids=(), **_):
        prompts.append(idx[0].tolist())
        for token in next(remaining)[:max_new_tokens]:
            yield torch.tensor([[token]])
            if token in stop_ids:
                return

    monkeypatch.setattr(engine.Leon, "stream", stream)
    return prompts


def call(text: str) -> list[int]:
    return [S["<|tool_call|>"], *TOKENIZER.encode(text), S["<|/tool_call|>"]]


def answer(text: str) -> list[int]:
    return [*TOKENIZER.encode(text), S["<|end|>"]]


def test_client_tools_return_tool_use_and_take_the_result(client, monkeypatch) -> None:
    prompts = script(monkeypatch, call('{"name": "tiempo", "arguments": {"ciudad": "Lima"}}'), answer("son 408"))
    question = [{"role": "user", "content": "hola"}]
    message = client.messages.create(model=MODEL, max_tokens=100, tools=[WEATHER], messages=question)
    assert message.stop_reason == "tool_use"
    use = next(block for block in message.content if block.type == "tool_use")
    assert use.name == "tiempo" and use.input == {"ciudad": "Lima"} and use.id.startswith("toolu_")
    # The tool is announced in the system prompt.
    assert "- tiempo(ciudad): el tiempo que hace" in TOKENIZER.decode(prompts[0])

    follow_up = [
        *question,
        {"role": "assistant", "content": [{"type": "tool_use", "id": use.id, "name": "tiempo", "input": use.input}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "soleado"}]},
    ]
    message = client.messages.create(model=MODEL, max_tokens=100, tools=[WEATHER], messages=follow_up)
    assert message.stop_reason == "end_turn" and message.content[-1].text == "son 408"
    # The answer continues the same turn: the prompt ends with the result, not a new <|assistant|>.
    assert TOKENIZER.decode(prompts[1]).endswith('"Lima"}}<|/tool_call|><|tool_result|>soleado<|/tool_result|>')
    count = client.messages.count_tokens(model=MODEL, tools=[WEATHER], messages=follow_up).input_tokens
    assert count == len(prompts[1])


def test_leon_tools_run_on_the_server_and_are_saved(app, client, monkeypatch) -> None:
    script(monkeypatch, call('{"name": "calculadora", "arguments": {"expresion": "17*24"}}'), answer("son 408"))
    web = app.test_client()
    conversation = web.post("/api/conversations", json={"mode": "chat", "model": MODEL}, headers=HOST).get_json()
    body = {
        "model": MODEL,
        "max_tokens": 100,
        "stream": True,
        "messages": [{"role": "user", "content": "¿17 por 24?"}],
        "leon": {"tools": True},
    }
    headers = {**HOST, "X-Leon-Conversation": conversation["id"]}
    events = [
        json.loads(line[6:])
        for line in web.post("/v1/messages", json=body, headers=headers).get_data(as_text=True).splitlines()
        if line.startswith("data: ")
    ]
    starts = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
    assert starts == ["text", "server_tool_use", "leon_tool_result", "text"]
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    assert {"type": "input_json_delta", "partial_json": '{"expresion": "17*24"}'} in deltas
    result = next(e["content_block"] for e in events if e.get("content_block", {}).get("type") == "leon_tool_result")
    assert result["content"] == "408"
    assert events[-2]["delta"]["stop_reason"] == "end_turn" and events[-2]["leon"]["tool_calls"] == 1

    saved = web.get(f"/api/conversations/{conversation['id']}", headers=HOST).get_json()["messages"]
    assert saved[1]["content"] == "son 408"
    assert saved[1]["steps"] == [
        {
            "id": saved[1]["steps"][0]["id"],
            "name": "calculadora",
            "input": {"expresion": "17*24"},
            "result": "408",
            "at": 0,
        }
    ]


def test_tool_requests_are_validated(client, app) -> None:
    web = app.test_client()
    tool_turns = [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "tiempo", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "sol"}]},
    ]
    cases = [
        ({"tool_choice": {"type": "any"}, "tools": [WEATHER]}, "tool_choice"),
        ({"messages": tool_turns}, "need `tools`"),
        ({"leon": {"tools": ["martillo"]}}, "unknown tool martillo"),
        ({"messages": tool_turns[:2], "tools": [WEATHER]}, "needs its tool_result"),
        (
            {
                "tools": [WEATHER],
                "messages": [
                    *tool_turns[:2],
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t1"}, {"type": "text", "text": "y"}],
                    },
                ],
            },
            "cannot be mixed",
        ),
    ]
    for extra, message in cases:
        body = {"model": MODEL, "max_tokens": 10, "messages": [{"role": "user", "content": "hola"}], **extra}
        response = web.post("/v1/messages", json=body, headers=HOST)
        assert response.status_code == 400 and message in response.get_json()["error"]["message"], (
            extra,
            response.get_json(),
        )


def test_context_meter_counts_the_tools_prompt(app) -> None:
    web = app.test_client()
    body = {"model": MODEL, "mode": "chat", "system": "", "messages": [], "draft": "hola"}
    plain = web.post("/api/context", json=body, headers=HOST).get_json()
    with_tools = web.post("/api/context", json={**body, "tools": True}, headers=HOST).get_json()
    assert with_tools["system"] > plain["system"] and with_tools["used"] > plain["used"]
