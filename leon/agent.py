"""The agent loop (phase 6): an agent is a loop, not a model.

    generate → the model writes <|tool_call|>{"name": …, "arguments": …}<|/tool_call|> and stops
    run      → the runtime parses the JSON and runs the tool (leon.tools)
    append   → <|tool_result|>result<|/tool_result|> goes into the context
    repeat   → the model continues the same answer, now knowing the result

until the model ends its turn (<|end|>), the token budget runs out or it has called too many tools.
The loop works on token ids and knows nothing about the model: `generate(ids, max_new, stop_ids)`
is any function that yields the next token ids (apps/api wraps Leon.stream with the user's
sampling settings; the tests use a scripted fake).
"""

import json
from collections.abc import Callable, Iterator

from leon.tools import TOOLS, Tool

MAX_TOOL_CALLS = 3

Generate = Callable[[list[int], int, set[int]], Iterator[int]]


def parse_call(text: str) -> dict:
    """{"name", "arguments"} from the text of a call, or ValueError with a message for the model."""
    try:
        call = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("la llamada no es JSON válido") from None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError('la llamada necesita "name"')
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError('"arguments" debe ser un objeto')
    return {"name": call["name"], "arguments": arguments}


def run_tool(call: dict, tools: dict[str, Tool]) -> str:
    tool = tools.get(call["name"])
    if tool is None:
        return f"error: no existe la herramienta {call['name']}"
    return tool(call["arguments"])


def run_agent(
    generate: Generate,
    tokenizer,
    ids: list[int],
    max_new_tokens: int,
    tools: dict[str, Tool] = TOOLS,
    max_calls: int = MAX_TOOL_CALLS,
) -> Iterator[dict]:
    """Run the loop from a prompt (encode_chat with the tools' system prompt), yielding events:

        {"text": piece}                      text of the answer, as it is generated
        {"tool_call": {"name", "arguments"}} a valid call (or {"raw", "error"} if it was not)
        {"tool_result": text}                what the tool returned (errors included)
        {"done": True, "stopped", "tokens", "tool_calls"}
            stopped: "end" | "max_tokens" | "max_tool_calls" | "tool_use"

    A tool without `run` belongs to the client (as with the Claude API's tools): the loop yields
    its call and stops with "tool_use"; the client runs it and sends the result in the next request.

    Text is only yielded once it is complete UTF-8. Tool results count towards neither the budget
    nor the generated tokens: the model did not write them.
    """
    special = tokenizer.special_tokens
    call_start, call_end = special["<|tool_call|>"], special["<|/tool_call|>"]
    stop_ids = set(special.values()) - {call_start}
    generated, calls = 0, 0
    while True:
        text_ids: list[int] = []
        emitted = ""
        call_ids: list[int] | None = None  # the call being written, once <|tool_call|> appears
        stopped = "max_tokens"
        for token in generate(ids, max_new_tokens - generated, stop_ids):
            generated += 1
            ids = [*ids, token]
            if token == call_start and call_ids is None:
                call_ids = []
                continue
            if token in stop_ids:
                stopped = "tool_call" if token == call_end and call_ids is not None else "end"
                break
            if call_ids is not None:
                call_ids.append(token)
                continue
            text_ids.append(token)
            text = tokenizer.decode(text_ids)
            if not text.endswith("�") and len(text) > len(emitted):
                yield {"text": text[len(emitted) :]}
                emitted = text
        text = tokenizer.decode(text_ids)
        if len(text) > len(emitted):
            yield {"text": text[len(emitted) :]}

        if stopped != "tool_call":
            yield {"done": True, "stopped": stopped, "tokens": generated, "tool_calls": calls}
            return
        if calls == max_calls:
            yield {"done": True, "stopped": "max_tool_calls", "tokens": generated, "tool_calls": calls}
            return
        calls += 1
        raw = tokenizer.decode(call_ids)
        try:
            call = parse_call(raw)
            tool = tools.get(call["name"])
            if tool is not None and tool.run is None:  # the client runs it
                yield {"tool_call": call}
                yield {"done": True, "stopped": "tool_use", "tokens": generated, "tool_calls": calls}
                return
            result = run_tool(call, tools)
            yield {"tool_call": call}
        except ValueError as error:
            result = f"error: {error}"
            yield {"tool_call": {"raw": raw, "error": str(error)}}
        yield {"tool_result": result}
        ids = [*ids, special["<|tool_result|>"], *tokenizer.encode(result), special["<|/tool_result|>"]]
        if generated >= max_new_tokens:
            yield {"done": True, "stopped": "max_tokens", "tokens": generated, "tool_calls": calls}
            return
