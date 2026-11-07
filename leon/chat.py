"""Chat template: how a conversation becomes a token sequence.

    <|system|>instructions<|end|><|user|>hello<|end|><|assistant|>answer<|end|><|user|>...<|end|><|assistant|>

From phase 6 an answer may call tools. The call is JSON between <|tool_call|> and <|/tool_call|>;
the runtime (leon.agent) runs the tool, appends its result between <|tool_result|> and
<|/tool_result|>, and the same assistant turn continues:

    <|user|>¿Cuánto es 17 por 24?<|end|><|assistant|><|tool_call|>{"name": "calculadora",
    "arguments": {"expresion": "17*24"}}<|/tool_call|><|tool_result|>408<|/tool_result|>Son 408.<|end|>

As messages: an assistant message with "tool_calls": [{"name", "arguments"}] (and optional text
before them), one {"role": "tool", "content": result} per call, then an assistant message with
the rest of the answer.

Phases 5 and 6 train the model on this exact format and apps/api uses it at inference time;
all of them import it from here so they can never drift apart.
"""

import json

from leon.tokenizer import BPETokenizer

ROLES = ("system", "user", "assistant", "tool")
# The system prompt of the SFT data (phase 5 gives it to a share of the conversations) and the
# chat app's default. apps/web/src/preferences.js repeats it (tests/test_chat.py checks they match).
DEFAULT_SYSTEM = "Eres Leon, un asistente que responde en español de forma clara y breve."


def tool_call_json(call: dict) -> str:
    """The text of a tool call, always written the same way: name first, then the arguments."""
    return json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}, ensure_ascii=False)


def _encode(tokenizer: BPETokenizer, messages: list[dict]) -> tuple[list[int], list[bool], bool]:
    """Token ids, which of them the model must learn to produce, and whether the last assistant
    turn is still open (it ended with tool calls or tool results, so it continues)."""
    special = tokenizer.special_tokens
    ids: list[int] = []
    learn: list[bool] = []
    turn_open = False  # inside an assistant turn that called tools and has not ended yet

    def add(tokens: list[int], learned: bool) -> None:
        ids.extend(tokens)
        learn.extend([learned] * len(tokens))

    for i, message in enumerate(messages):
        role = message["role"]
        if role not in ROLES:
            raise ValueError(f"invalid role: {role}")
        if role == "tool":
            previous = messages[i - 1]["role"] if i else None
            if not turn_open or previous not in ("assistant", "tool"):
                raise ValueError("a tool result must follow an assistant message with tool_calls")
            # Written by the runtime, never by the model: context only.
            add([special["<|tool_result|>"], *tokenizer.encode(message["content"]), special["<|/tool_result|>"]], False)
            continue
        if role == "assistant":
            if not turn_open:
                add([special["<|assistant|>"]], False)  # the turn marker is context: the prompt ends with it
            add(tokenizer.encode(message.get("content") or ""), True)
            calls = message.get("tool_calls") or []
            for call in calls:
                add(
                    [special["<|tool_call|>"], *tokenizer.encode(tool_call_json(call)), special["<|/tool_call|>"]], True
                )
            turn_open = bool(calls)
            if not calls:
                add([special["<|end|>"]], True)  # learned: the model must know when to stop
            continue
        if turn_open:
            raise ValueError(f"the assistant turn that called tools must end before a {role} message")
        add([special[f"<|{role}|>"], *tokenizer.encode(message["content"]), special["<|end|>"]], False)
    return ids, learn, turn_open


def encode_chat(
    tokenizer: BPETokenizer,
    messages: list[dict],
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> list[int]:
    """Encode [{"role": ..., "content": ...}, ...] with the chat template.

    Control tokens are inserted by id and message text is encoded as ordinary text, so a
    message containing "<|end|><|assistant|>" cannot forge a turn (prompt injection by tokens).
    With add_generation_prompt, a final <|assistant|> invites the model to answer; after tool
    results nothing is added, because the model continues the same turn.

    With continue_final_message, the last message must be the assistant's and is left open
    (no <|end|>, no generation prompt): the model continues that text ("prefill").
    """
    if continue_final_message and (not messages or messages[-1]["role"] != "assistant"):
        raise ValueError("continue_final_message needs the assistant's message last")
    if continue_final_message and messages[-1].get("tool_calls"):
        raise ValueError("continue_final_message cannot continue a message that ends in tool calls")
    ids, _, turn_open = _encode(tokenizer, messages)
    if continue_final_message:
        return ids[:-1]  # drop the final <|end|>
    if add_generation_prompt and not turn_open:
        ids.append(tokenizer.special_tokens["<|assistant|>"])
    return ids


def encode_chat_for_training(tokenizer: BPETokenizer, messages: list[dict]) -> tuple[list[int], list[bool]]:
    """Encode a whole conversation for supervised fine-tuning (phases 5 and 6).

    Returns the token ids and, for each of them, whether the model must learn to *produce* it.
    Only the assistant's answers are learned: their text, their tool calls (markers and JSON)
    and their closing <|end|> (so the model learns when to stop). The system and user turns,
    the <|assistant|> marker that opens each answer and the tool results are context: the model
    reads them but is never trained to write them (loss masking).
    The sequence is exactly what encode_chat produces with add_generation_prompt=False.
    """
    ids, learn, _ = _encode(tokenizer, messages)
    return ids, learn
