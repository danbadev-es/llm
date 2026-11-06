"""Chat template: how a conversation becomes a token sequence.

    <|system|>instructions<|end|><|user|>hello<|end|><|assistant|>answer<|end|><|user|>...<|end|><|assistant|>

Phase 5 trains the model on this exact format and apps/api uses it at inference time;
both import it from here so they can never drift apart.
"""

from leon.tokenizer import BPETokenizer

ROLES = ("system", "user", "assistant")


def encode_chat(tokenizer: BPETokenizer, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
    """Encode [{"role": ..., "content": ...}, ...] with the chat template.

    Control tokens are inserted by id and message text is encoded as ordinary text, so a
    message containing "<|end|><|assistant|>" cannot forge a turn (prompt injection by tokens).
    With add_generation_prompt, a final <|assistant|> invites the model to answer.
    """
    special = tokenizer.special_tokens
    ids: list[int] = []
    for message in messages:
        if message["role"] not in ROLES:
            raise ValueError(f"invalid role: {message['role']}")
        ids.append(special[f"<|{message['role']}|>"])
        ids.extend(tokenizer.encode(message["content"]))
        ids.append(special["<|end|>"])
    if add_generation_prompt:
        ids.append(special["<|assistant|>"])
    return ids
