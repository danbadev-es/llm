"""Model side of the chat: find checkpoints, build prompts, stream generated text.

Kept free of any web framework so it can be reused. Every generation endpoint (the
Anthropic-compatible /v1 API in anthropic_api.py) turns its request into the same `params`
dict and calls Engine.stream():

    model, mode ("chat" | "complete"), temperature, top_k, top_p, max_new_tokens, stop_sequences
    complete mode: prompt                 the raw text to continue
    chat mode:     system, messages       and prefill: continue the final assistant message
                   tools (optional)       {name: leon.tools.Tool}: the answer runs the agent loop

With tools, messages may hold tool calls and results (leon.chat), the system prompt lists the
tools, and the stream also yields {"tool_call", "id"} and {"tool_result", "id"} events. Tools
with `run` (Leon's own) are run here; a call to a client tool ends the answer ("tool_use").
"""

import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import torch

from leon.agent import run_agent
from leon.chat import DEFAULT_SYSTEM, encode_chat
from leon.data import load_tokenizer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR, LEGACY_TOKENIZER
from leon.tools import TOOLS, system_prompt

MODES = ("complete", "chat")


def find_checkpoints() -> dict[str, Path]:
    """Every Leon checkpoint: checkpoints/<N>_*/<run>/best.pt, for N >= 3."""
    paths = sorted(CHECKPOINT_DIR.glob("[3-9]_*/*/best.pt"))
    return {f"{p.parent.parent.name}/{p.parent.name}": p for p in paths}


def effective_system(params: dict) -> str:
    """The system prompt sent to the model: with tools, the tools' list after the instructions
    (Leon's default ones if there are none, as in the phase 6 data)."""
    tools = params.get("tools")
    if not tools:
        return params["system"]
    return system_prompt(list(tools.values()), base=params["system"] or DEFAULT_SYSTEM)


def read_number(request: dict, key: str, default, low, high, integer: bool = False):
    """Read an optional numeric field, rejecting wrong types and values outside [low, high]."""
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (integer and not isinstance(value, int)):
        raise ValueError(f"'{key}' must be {'an integer' if integer else 'a number'}")
    if not (low <= value <= high):  # also rejects NaN, which compares False with everything
        raise ValueError(f"'{key}' must be between {low} and {high}")
    return value


def _find_stop(text: str, stop_sequences: list[str]) -> tuple[int, str | None]:
    """Where the earliest stop sequence starts in text, and which one: (len(text), None) if none."""
    hits = [(text.find(stop), stop) for stop in stop_sequences if stop in text]
    return min(hits) if hits else (len(text), None)


def _partial_stop(text: str, stop_sequences: list[str]) -> int:
    """Length of the longest end of text that could still grow into a stop sequence."""
    return max((k for stop in stop_sequences for k in range(1, len(stop)) if text.endswith(stop[:k])), default=0)


class Engine:
    """Keeps one checkpoint in memory and serves one generation at a time."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.tokenizers: dict = {}  # loaded once per version: every checkpoint says which one it uses
        self.lock = threading.Lock()
        self.name: str | None = None
        self.version: float | None = None  # modification time of the loaded checkpoint file
        self.model: Leon | None = None
        self.use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def list_models(self) -> list[dict]:
        models = []
        for name, path in find_checkpoints().items():
            # mmap: reads only the metadata, not the hundreds of MB of weights
            ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            config = LeonConfig(**ckpt["config"])
            models.append(
                {
                    "name": name,
                    "stage": ckpt.get("stage", "base"),  # base (3-4), chat (5), agent (6-8: tools)
                    "params": sum(t.numel() for k, t in ckpt["model"].items() if k != "lm_head.weight"),  # tied
                    "context": config.block_size,
                    "max_new_tokens": config.block_size // 2,  # the rest of the context is for the prompt
                    "step": ckpt["step"],
                }
            )
        return models

    def _load(self, name: str) -> Leon:
        """Load a checkpoint, or reuse the one in memory if neither the name nor the file changed.

        A run that is still training rewrites its best.pt: the newer weights are picked up
        on the next request, so the chat can follow a model while it learns.
        """
        path = find_checkpoints()[name]
        version = path.stat().st_mtime
        if name != self.name or version != self.version:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            model = Leon(LeonConfig(**ckpt["config"])).to(self.device).eval()
            model.load_state_dict(ckpt["model"])
            self.model, self.name, self.version = model, name, version  # the previous model is freed
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return self.model

    def tokenizer_for(self, name: str):
        """The tokenizer a checkpoint was trained with (its "tokenizer" key; v1 if it has none)."""
        meta = torch.load(find_checkpoints()[name], map_location="cpu", weights_only=True, mmap=True)
        version = meta.get("tokenizer", LEGACY_TOKENIZER)
        if version not in self.tokenizers:
            self.tokenizers[version] = load_tokenizer(version)
        return self.tokenizers[version]

    def _chat_prompt(self, tokenizer, params: dict, budget: int) -> tuple[list[int], int]:
        """Encode the conversation within `budget` tokens.

        When it does not fit, whole turns are dropped starting from the oldest, always keeping
        the system instructions and the last message. Returns (ids, number of dropped turns).
        """
        text = effective_system(params)
        system = [{"role": "system", "content": text}] if text else []
        messages = params["messages"]
        for dropped in range(len(messages)):
            if messages[dropped]["role"] == "tool":
                continue  # a tool result cannot start the prompt without its call
            try:
                ids = encode_chat(tokenizer, system + messages[dropped:], continue_final_message=params["prefill"])
            except ValueError:
                continue  # it would start inside an assistant turn that called tools
            if len(ids) <= budget or dropped == len(messages) - 1:
                return ids[-budget:], dropped  # only a single huge message gets cut from its start
        # Only a tool exchange that does not fit at all is left: keep its end.
        return encode_chat(tokenizer, system + messages, continue_final_message=params["prefill"])[-budget:], 0

    def context_usage(
        self, model: str, mode: str, system: str, messages: list[dict], draft: str, max_new: int, tools: bool = False
    ) -> dict:
        """How the next request would fill the model's context, for the meter in the web app.

        The same arithmetic as stream(): the answer reserves min(max_new, context / 2) tokens and
        the prompt gets the rest. In chat mode the prompt is the system instructions, the history
        and the draft as a new user message; whole turns that do not fit are dropped, oldest
        first (there is no summarizing "compaction": a small model would summarize badly).
        In complete mode only the draft is sent, and its beginning is cut if it is too long.
        With tools, the system instructions include the list of Leon's tools.
        """
        meta = torch.load(find_checkpoints()[model], map_location="cpu", weights_only=True, mmap=True)
        context = meta["config"]["block_size"]
        reserved = min(max_new, context // 2)
        budget = context - reserved
        tokenizer = self.tokenizer_for(model)
        usage = {"context": context, "reserved": reserved, "system": 0, "history": 0, "draft": 0}
        if mode == "complete":
            total = len(tokenizer.encode(draft))
            usage["draft"] = min(total, budget)
            return {**usage, "used": usage["draft"], "free": budget - usage["draft"], "total": total,
                    "dropped_messages": 0, "dropped_tokens": total - usage["draft"]}  # fmt: skip
        system = effective_system({"system": system, "tools": TOOLS if tools else None})
        system_turn = [{"role": "system", "content": system}] if system else []
        turns = messages + ([{"role": "user", "content": draft}] if draft else [])
        if system_turn:
            usage["system"] = len(encode_chat(tokenizer, system_turn, add_generation_prompt=False))
        if not turns:
            return {**usage, "used": usage["system"], "free": budget - usage["system"], "total": usage["system"],
                    "dropped_messages": 0, "dropped_tokens": 0}  # fmt: skip
        total = len(encode_chat(tokenizer, system_turn + turns))
        ids, dropped = self._chat_prompt(tokenizer, {"system": system, "messages": turns, "prefill": False}, budget)
        if draft:  # the new user turn plus the <|assistant|> that opens the answer
            usage["draft"] = len(encode_chat(tokenizer, turns[-1:]))
        else:
            usage["draft"] = 1 if messages else 0  # just the <|assistant|> marker
        usage["draft"] = min(usage["draft"], len(ids))
        usage["system"] = min(usage["system"], len(ids) - usage["draft"])
        usage["history"] = len(ids) - usage["draft"] - usage["system"]
        return {**usage, "used": len(ids), "free": budget - len(ids), "total": total,
                "dropped_messages": dropped, "dropped_tokens": total - len(ids)}  # fmt: skip

    def count_tokens(self, params: dict) -> int:
        """Prompt length in tokens, before any old turn is dropped to make it fit."""
        tokenizer = self.tokenizer_for(params["model"])
        if params["mode"] == "complete":
            return len(tokenizer.encode(params["prompt"]))
        text = effective_system(params)
        system = [{"role": "system", "content": text}] if text else []
        return len(encode_chat(tokenizer, system + params["messages"], continue_final_message=params["prefill"]))

    def stream(self, params: dict) -> Iterator[dict]:
        """Yield one {"started": ...} event, {"text": piece} events while generating, then {"done": ...}.

        The final event carries the stats: tokens, tok_per_s, prompt_tokens, dropped_turns,
        stopped ("end" | "max_tokens" | "stop_sequence") and stop_sequence (the one that matched).
        Text that matches a stop sequence is never sent, and text that could still become one
        is held back until the next token decides it. Closing the generator (the client
        disconnected) stops generation and releases the lock.
        """
        with self.lock:
            model = self._load(params["model"])
            tokenizer = self.tokenizer_for(params["model"])
            block_size = model.config.block_size
            max_new = min(params["max_new_tokens"], block_size // 2)
            budget = block_size - max_new  # prompt tokens that leave room for the answer

            dropped_turns = 0
            if params["mode"] == "chat":
                ids, dropped_turns = self._chat_prompt(tokenizer, params, budget)
                # Any control token ends the answer: normally <|end|>, but also <|user|>,
                # <|endoftext|>... if the model tries to start another turn or document.
                stop_ids = set(tokenizer.special_tokens.values())
            else:
                ids = tokenizer.encode(params["prompt"])[-budget:]  # keep the most recent text
                stop_ids = {tokenizer.eot_id}

            # The prompt is ready (and the model loaded): endpoints can start their response.
            yield {"started": True, "prompt_tokens": len(ids), "dropped_turns": dropped_turns}
            if params["mode"] == "chat" and params.get("tools"):
                yield from self._agent(model, tokenizer, params, ids, max_new, dropped_turns)
                return
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            stops = params["stop_sequences"]
            generated: list[int] = []
            emitted = ""
            stopped, stop_sequence = "max_tokens", None
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                tokens = model.stream(
                    idx,
                    max_new,
                    temperature=params["temperature"],
                    top_k=params["top_k"] or None,
                    top_p=params["top_p"],
                    stop_ids=stop_ids,
                )
                for next_id in tokens:
                    token = next_id.item()
                    if token in stop_ids:
                        stopped = "end"
                        break
                    generated.append(token)
                    # A token may end in the middle of a multi-byte character: wait for the rest.
                    text = tokenizer.decode(generated)
                    if text.endswith("�"):
                        continue
                    cut, stop_sequence = _find_stop(text, stops)
                    if stop_sequence is not None:
                        break
                    cut -= _partial_stop(text, stops)
                    if cut > len(emitted):
                        yield {"text": text[len(emitted) : cut]}
                        emitted = text[:cut]
            # The rest: held-back text, or a character still incomplete when generation ended.
            text = tokenizer.decode(generated)
            cut, stop_sequence = _find_stop(text, stops)
            if stop_sequence is not None:
                stopped = "stop_sequence"
            if cut > len(emitted):
                yield {"text": text[len(emitted) : cut]}
            elapsed = time.perf_counter() - start
            yield {
                "done": True,
                "tokens": len(generated),
                "tok_per_s": len(generated) / elapsed if elapsed > 0 else 0.0,
                "prompt_tokens": len(ids),
                "dropped_turns": dropped_turns,
                "stopped": stopped,
                "stop_sequence": stop_sequence,
            }

    def _agent(self, model: Leon, tokenizer, params: dict, ids: list[int], max_new: int, dropped: int):
        """The answer through the agent loop (leon.agent): text, tool calls and their results.

        Each call gets an id (toolu_…) shared by its result. Stop sequences do not apply here:
        the answer may be split by tool calls. Runs with the engine's lock held (stream()).
        """
        block_size = model.config.block_size

        def generate(context: list[int], budget: int, stop_ids: set[int]):
            # The context grows with every result: keep its end (stream() crops it the same way).
            idx = torch.tensor([context[-block_size:]], dtype=torch.long, device=self.device)
            for token in model.stream(
                idx,
                budget,
                temperature=params["temperature"],
                top_k=params["top_k"] or None,
                top_p=params["top_p"],
                stop_ids=stop_ids,
            ):
                yield token.item()

        start = time.perf_counter()
        call_id = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            for event in run_agent(generate, tokenizer, ids, max_new, tools=params["tools"]):
                if "tool_call" in event:
                    call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    client = event["tool_call"].get("name") in params["tools"] and (
                        params["tools"][event["tool_call"]["name"]].run is None
                    )
                    yield {"tool_call": event["tool_call"], "id": call_id, "client": client}
                elif "tool_result" in event:
                    yield {"tool_result": event["tool_result"], "id": call_id}
                elif "text" in event:
                    yield event
                else:
                    elapsed = time.perf_counter() - start
                    yield {
                        "done": True,
                        "tokens": event["tokens"],
                        "tok_per_s": event["tokens"] / elapsed if elapsed > 0 else 0.0,
                        "prompt_tokens": len(ids),
                        "dropped_turns": dropped,
                        "stopped": "end" if event["stopped"] == "max_tool_calls" else event["stopped"],
                        "stop_sequence": None,
                        "tool_calls": event["tool_calls"],
                    }
