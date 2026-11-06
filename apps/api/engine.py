"""Model side of the chat: find checkpoints, build prompts, stream generated text.

Kept free of any web framework so it can be reused (phase 6 will add the tool-calling loop here).
"""

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from leon.chat import encode_chat
from leon.data import load_tokenizer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR

MODES = ("complete", "chat")


def find_checkpoints() -> dict[str, Path]:
    """Every Leon checkpoint: checkpoints/<N>_*/<run>/best.pt, for N >= 3."""
    paths = sorted(CHECKPOINT_DIR.glob("[3-9]_*/*/best.pt"))
    return {f"{p.parent.parent.name}/{p.parent.name}": p for p in paths}


def _number(request: dict, key: str, default, low, high, integer: bool = False):
    """Read an optional numeric field, rejecting wrong types and values outside [low, high]."""
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (integer and not isinstance(value, int)):
        raise ValueError(f"'{key}' must be {'an integer' if integer else 'a number'}")
    if not (low <= value <= high):  # also rejects NaN, which compares False with everything
        raise ValueError(f"'{key}' must be between {low} and {high}")
    return value


def validate(request) -> dict:
    """Check a /api/generate body and return its normalized parameters.

    Everything is checked here, before any response header is sent, so a bad request gets
    a clean 400 instead of failing in the middle of the event stream.
    """
    if not isinstance(request, dict):
        raise ValueError("the body must be a JSON object")
    if request.get("model") not in find_checkpoints():
        raise ValueError(f"unknown model: {request.get('model')}")
    if request.get("mode") not in MODES:
        raise ValueError(f"'mode' must be one of {MODES}")
    params = {
        "model": request["model"],
        "mode": request["mode"],
        "temperature": float(_number(request, "temperature", 0.8, 0.0, 5.0)),
        "top_k": _number(request, "top_k", 40, 0, 100_000, integer=True),  # 0 = no top-k filter
        "max_new_tokens": _number(request, "max_new_tokens", 200, 1, 100_000, integer=True),
    }
    if request["mode"] == "complete":
        if not isinstance(request.get("prompt"), str) or not request["prompt"]:
            raise ValueError("complete mode needs a non-empty string 'prompt'")
        params["prompt"] = request["prompt"]
        return params

    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("chat mode needs a non-empty 'messages' list")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("user", "assistant")
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("each message needs 'role' ('user' or 'assistant') and a string 'content'")
    system = request.get("system", "")
    if not isinstance(system, str):
        raise ValueError("'system' must be a string")
    params["messages"] = [{"role": m["role"], "content": m["content"]} for m in messages]
    params["system"] = system.strip()
    return params


class Engine:
    """Keeps one checkpoint in memory and serves one generation at a time."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.tokenizer = load_tokenizer()
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
                    "stage": ckpt.get("stage", "base"),  # phase 5 checkpoints will be saved with stage="chat"
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

    def _chat_prompt(self, params: dict, budget: int) -> tuple[list[int], int]:
        """Encode the conversation within `budget` tokens.

        When it does not fit, whole turns are dropped starting from the oldest, always keeping
        the system instructions and the last message. Returns (ids, number of dropped turns).
        """
        system = [{"role": "system", "content": params["system"]}] if params["system"] else []
        messages = params["messages"]
        for dropped in range(len(messages)):
            ids = encode_chat(self.tokenizer, system + messages[dropped:])
            if len(ids) <= budget or dropped == len(messages) - 1:
                return ids[-budget:], dropped  # only a single huge message gets cut from its start
        raise AssertionError("unreachable: validate() guarantees at least one message")

    def stream(self, params: dict) -> Iterator[dict]:
        """Yield {"text": piece} events while generating, then one {"done": ...} event.

        `params` comes from validate(). Closing the generator (the client disconnected)
        stops generation and releases the lock.
        """
        with self.lock:
            model = self._load(params["model"])
            block_size = model.config.block_size
            max_new = min(params["max_new_tokens"], block_size // 2)
            budget = block_size - max_new  # prompt tokens that leave room for the answer

            dropped_turns = 0
            if params["mode"] == "chat":
                ids, dropped_turns = self._chat_prompt(params, budget)
                # Any control token ends the answer: normally <|end|>, but also <|user|>,
                # <|endoftext|>... if the model tries to start another turn or document.
                stop_ids = set(self.tokenizer.special_tokens.values())
            else:
                ids = self.tokenizer.encode(params["prompt"])[-budget:]  # keep the most recent text
                stop_ids = {self.tokenizer.eot_id}

            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            generated: list[int] = []
            emitted = ""
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                tokens = model.stream(
                    idx, max_new, temperature=params["temperature"], top_k=params["top_k"] or None, stop_ids=stop_ids
                )
                for next_id in tokens:
                    token = next_id.item()
                    if token in stop_ids:
                        break
                    generated.append(token)
                    # A token may end in the middle of a multi-byte character: wait for the rest.
                    text = self.tokenizer.decode(generated)
                    if text.endswith("�"):
                        continue
                    yield {"text": text[len(emitted) :]}
                    emitted = text
            tail = self.tokenizer.decode(generated)[len(emitted) :]
            if tail:  # a character still incomplete when generation ended is sent as it is
                yield {"text": tail}
            elapsed = time.perf_counter() - start
            yield {
                "done": True,
                "tokens": len(generated),
                "tok_per_s": len(generated) / elapsed if elapsed > 0 else 0.0,
                "prompt_tokens": len(ids),
                "dropped_turns": dropped_turns,
                "stopped": "end" if len(generated) < max_new else "max_tokens",
            }
