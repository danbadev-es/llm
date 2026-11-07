"""Phase 8: arithmetic problems with step-by-step solutions (CPU, ~1 min).

Five levels of difficulty, each problem written twice:

    steps    the solution, one small operation per line, then "Respuesta: N"
    direct   only "Respuesta: N" (the control: the same problems, no room to think)

    1  two-digit sums                      47 + 38
    2  two digits by one                   47 × 6
    3  two digits by two                   47 × 36
    4  mixed, with parentheses             (47 + 38) × 6, 47 × 6 − 38
    5  word problems (two operations)      "Compro 6 cajas de 47 lápices y regalo 38..."

Every step is an operation a small model can learn to do by heart: one-digit products, sums of
round numbers. A 47 × 36 that Leon cannot do in one token becomes 8 easy ones.

Output in data/reasoning/: {steps,direct}/{train,val}.{jsonl,npz} (the SFT format of phase 5,
with a share of phase 5 conversations so the model keeps chatting), problems.jsonl (problems
and answers for GRPO) and eval.jsonl (held-out problems by level, for evaluate.py).
"""

import argparse
import hashlib
import json
import random

from leon.chat import encode_chat_for_training
from leon.data import load_tokenizer, save_conversations
from leon.paths import DATA_DIR, DEFAULT_TOKENIZER

REASONING_DIR = DATA_DIR / "reasoning"
LEVELS = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-level", type=int, default=4000, help="training problems per level")
    parser.add_argument("--grpo", type=int, default=5000, help="problems (levels 2-5) for GRPO, with their answers")
    parser.add_argument("--eval", type=int, default=100, help="held-out problems per level")
    parser.add_argument("--chat", type=int, default=3000, help="phase 5 conversations mixed in")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def times_one(a: int, factor: int) -> str:
    """a × factor (factor of one significant digit: 7 or 30), splitting a into tens and units."""
    tens, units = a // 10 * 10, a % 10
    if units == 0 or a < 10:
        return f"{a} × {factor} = {a * factor}"
    return (
        f"{a} × {factor} = {tens} × {factor} + {units} × {factor} = {tens * factor} + {units * factor} = {a * factor}"
    )


def multiply_steps(a: int, b: int) -> tuple[list[str], int]:
    """a × b by place value: each partial product is a one-digit product with zeros."""
    if b < 10:
        return [times_one(a, b)], a * b
    b_tens, b_units = b // 10 * 10, b % 10
    if b_units == 0:
        return [times_one(a, b)], a * b
    lines = [f"{a} × {b} = {a} × {b_tens} + {a} × {b_units}", times_one(a, b_tens), times_one(a, b_units)]
    lines.append(f"{a * b_tens} + {a * b_units} = {a * b}")
    return lines, a * b


def add_steps(a: int, b: int) -> tuple[list[str], int]:
    """a + b by place value: tens with tens, units with units."""
    a_tens, b_tens = a // 10 * 10, b // 10 * 10
    lines = [
        f"Decenas: {a_tens} + {b_tens} = {a_tens + b_tens}",
        f"Unidades: {a % 10} + {b % 10} = {a % 10 + b % 10}",
        f"{a_tens + b_tens} + {a % 10 + b % 10} = {a + b}",
    ]
    return lines, a + b


def subtract_steps(a: int, b: int) -> tuple[list[str], int]:
    """a − b in two moves: first the round part of b, then the rest."""
    round_part, rest = b // 10 * 10, b % 10
    lines = [f"{a} − {round_part} = {a - round_part}", f"{a - round_part} − {rest} = {a - b}"]
    return lines, a - b


ITEMS = [  # container (feminine: "cada una", "en las"), thing, "cuántos" or "cuántas"
    ("cajas", "lápices", "cuántos"), ("bolsas", "manzanas", "cuántas"), ("filas", "sillas", "cuántas"),
    ("cestas", "galletas", "cuántas"), ("estanterías", "libros", "cuántos"),
]  # fmt: skip


def problem(level: int, rng: random.Random) -> dict:
    """{"level", "question", "steps": [lines], "answer"}."""
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    op_word = rng.choice(["×", "por"])
    if level == 1:
        steps, answer = add_steps(a, b)
        question = rng.choice(["¿Cuánto es {a} + {b}?", "¿Cuánto es {a} más {b}?", "Calcula {a} + {b}."]).format(
            a=a, b=b
        )
    elif level == 2:
        b = rng.randint(2, 9)
        steps, answer = multiply_steps(a, b)
        question = f"¿Cuánto es {a} {op_word} {b}?"
    elif level == 3:
        steps, answer = multiply_steps(a, b)
        question = rng.choice([f"¿Cuánto es {a} {op_word} {b}?", f"Calcula {a} × {b}.", f"Multiplica {a} por {b}."])
    elif level == 4:
        c = rng.randint(2, 9)
        if rng.random() < 0.5:
            first, total = add_steps(a, b)
            second, answer = multiply_steps(total, c)
            steps = [f"Primero el paréntesis: {a} + {b}", *first, f"Después: {total} × {c}", *second]
            question = f"¿Cuánto es ({a} + {b}) × {c}?"
        else:
            first, product = multiply_steps(a, c)
            second, answer = subtract_steps(product, b)
            steps = [f"Primero la multiplicación: {a} × {c}", *first, f"Después: {product} − {b}", *second]
            question = f"¿Cuánto es {a} × {c} − {b}?"
    else:
        c = rng.randint(2, 9)
        container, thing, how_many = rng.choice(ITEMS)
        ask = how_many.capitalize()
        first, product = multiply_steps(a, c)
        if rng.random() < 0.5 and b < product:
            second, answer = subtract_steps(product, b)
            question = f"Tengo {c} {container} con {a} {thing} cada una y regalo {b} {thing}. ¿{ask} {thing} me quedan?"
            steps = [f"{thing.capitalize()} en total: {c} × {a}", *first, f"Me quedan: {product} − {b}", *second]
        else:
            second, answer = add_steps(product, b)
            question = f"Tengo {c} {container} con {a} {thing} cada una y compro {b} {thing} más. ¿{ask} {thing} tengo?"
            steps = [
                f"{thing.capitalize()} en las {container}: {c} × {a}",
                *first,
                f"En total: {product} + {b}",
                *second,
            ]
    return {"level": level, "question": question, "steps": steps, "answer": answer}


def conversation(item: dict, style: str) -> list[dict]:
    answer = f"Respuesta: {item['answer']}"
    body = "\n".join(item["steps"]) + "\n" + answer if style == "steps" else answer
    return [{"role": "user", "content": item["question"]}, {"role": "assistant", "content": body}]


def stable_fraction(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def chat_conversations(count: int, rng: random.Random) -> list[list[dict]]:
    """Phase 5 conversations without arithmetic, so the model keeps chatting."""
    items = []
    with open(DATA_DIR / "sft" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            messages = json.loads(line)["messages"]
            text = " ".join(m["content"] for m in messages)
            if len(text) < 2000 and not any(
                ch.isdigit() for ch in messages[1 if messages[0]["role"] == "system" else 0]["content"]
            ):
                items.append(messages)
    rng.shuffle(items)
    return items[:count]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tokenizer = load_tokenizer()
    held_out: set[str] = set()

    def unseen(level: int, unique: bool = False) -> dict:
        """A problem that is not in the evaluation set. Training may repeat a problem (level 2 has
        only ~1,400 different ones), but never one the evaluation will ask."""
        while True:
            item = problem(level, rng)
            if item["question"] not in held_out:
                if unique:
                    held_out.add(item["question"])
                return item

    evaluation = [unseen(level, unique=True) for level in LEVELS for _ in range(args.eval)]
    training = [unseen(level) for level in LEVELS for _ in range(args.per_level)]
    grpo = [unseen(rng.choice(LEVELS[1:])) for _ in range(args.grpo)]
    chat = chat_conversations(args.chat, rng)

    REASONING_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"tokenizer": DEFAULT_TOKENIZER}
    for style in ("steps", "direct"):
        splits = {"train": [], "val": []}
        for item in training:
            split = "val" if stable_fraction(item["question"]) < 0.05 else "train"
            splits[split].append((f"nivel_{item['level']}", conversation(item, style)))
        splits["train"] += [("chat", messages) for messages in chat]
        stats[style] = {}
        for split, items in splits.items():
            rng.shuffle(items)
            folder = REASONING_DIR / style
            folder.mkdir(exist_ok=True)
            with open(folder / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for source, messages in items:
                    f.write(json.dumps({"source": source, "messages": messages}, ensure_ascii=False) + "\n")
            encoded = [encode_chat_for_training(tokenizer, messages) for _, messages in items]
            stats[style][split] = save_conversations(
                folder / f"{split}.npz", encoded, [source for source, _ in items], DEFAULT_TOKENIZER
            )
        print(f"{style}: {stats[style]}", flush=True)
    for name, items in (("problems", grpo), ("eval", evaluation)):
        with open(REASONING_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    stats["grpo_problems"], stats["eval_problems"] = len(grpo), len(evaluation)
    (REASONING_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    example = training[2 * args.per_level]  # a level 3 problem
    print(
        f"\nExample (level 3):\n{example['question']}\n"
        + "\n".join(example["steps"])
        + f"\nRespuesta: {example['answer']}"
    )


if __name__ == "__main__":
    main()
