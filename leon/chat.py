"""Chat template: how a conversation becomes a token sequence.

    <|system|>instructions<|end|><|user|>hello<|end|><|assistant|>answer<|end|><|user|>...<|end|><|assistant|>

Phase 5 trains the model on this exact format and apps/api uses it at inference time;
both import it from here so they can never drift apart.
"""

from leon.tokenizer import BPETokenizer

ROLES = ("system", "user", "assistant")


def encode_chat(
    tokenizer: BPETokenizer,
    messages: list[dict],
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> list[int]:
    """Encode [{"role": ..., "content": ...}, ...] with the chat template.

    Control tokens are inserted by id and message text is encoded as ordinary text, so a
    message containing "<|end|><|assistant|>" cannot forge a turn (prompt injection by tokens).
    With add_generation_prompt, a final <|assistant|> invites the model to answer.

    With continue_final_message, the last message must be the assistant's and is left open
    (no <|end|>, no generation prompt): the model continues that text ("prefill").
    """
    if continue_final_message and (not messages or messages[-1]["role"] != "assistant"):
        raise ValueError("continue_final_message needs the assistant's message last")
    special = tokenizer.special_tokens
    ids: list[int] = []
    for message in messages:
        if message["role"] not in ROLES:
            raise ValueError(f"invalid role: {message['role']}")
        ids.append(special[f"<|{message['role']}|>"])
        ids.extend(tokenizer.encode(message["content"]))
        ids.append(special["<|end|>"])
    if continue_final_message:
        return ids[:-1]
    if add_generation_prompt:
        ids.append(special["<|assistant|>"])
    return ids


def encode_chat_for_training(tokenizer: BPETokenizer, messages: list[dict]) -> tuple[list[int], list[bool]]:
    """Encode a whole conversation for supervised fine-tuning (phase 5).

    Returns the token ids and, for each of them, whether the model must learn to *produce* it.
    Only the assistant's answers are learned, including their closing <|end|> (so the model learns
    when to stop). The system and user turns, and the <|assistant|> marker that opens each answer,
    are context: the model reads them but is never trained to write them (loss masking).
    The sequence is exactly what encode_chat produces with add_generation_prompt=False.
    """
    special = tokenizer.special_tokens
    ids: list[int] = []
    learn: list[bool] = []
    for message in messages:
        if message["role"] not in ROLES:
            raise ValueError(f"invalid role: {message['role']}")
        body = tokenizer.encode(message["content"]) + [special["<|end|>"]]
        is_answer = message["role"] == "assistant"
        ids += [special[f"<|{message['role']}|>"], *body]
        learn += [False] + [is_answer] * len(body)
    return ids, learn
