"""Phase 6: build the tool-calling dataset (CPU only, a few minutes).

There is no Spanish tool-calling dataset for Leon's three tools, so the dialogs are generated
from templates, with the real tools producing every result (the model learns from exactly what
the runtime will return). Each conversation has the tools' system prompt.

    calculadora      arithmetic questions in many phrasings, percentages, word problems
    fecha_hora       date, time, year, weekday, month
    documentos       questions about Wikipedia articles, answered from the search result
    no_encontrado    searches that find nothing: Leon must say so, not invent
    varias           two calls in one answer (date + arithmetic, chained arithmetic)
    sin_herramienta  phase 5 conversations (with the tools' system prompt): when not to call

Documents: Wikipedia *validation* articles, never seen in training. 300 go to data/docs/ejemplo/
(the demo documents of the chat app and the evaluation); another 2,500, kept apart in
data/agent/docs_pool/, are the ones the training dialogs search, so the evaluation measures
searching documents the model has never seen.

Output in data/agent/: {train,val}.jsonl and .npz (the format of phase 5), eval.jsonl
(held-out questions with the expected tool and answer, for evaluate.py) and stats.json.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from leon.chat import encode_chat_for_training
from leon.data import load_tokenizer, save_conversations
from leon.paths import DATA_DIR, DEFAULT_TOKENIZER
from leon.tools import (
    DAYS,
    MONTHS,
    calculate,
    format_datetime,
    search_documents,
    system_prompt,
)

AGENT_DIR = DATA_DIR / "agent"
POOL_DIR = AGENT_DIR / "docs_pool"
DEMO_DIR = DATA_DIR / "docs" / "ejemplo"
WIKI_VAL = DATA_DIR / "wiki_es" / "val.jsonl"
SYSTEM = system_prompt()
END_SECTIONS = re.compile(r"\n\n(Referencias|Enlaces externos|Véase también|Bibliografía|Notas)\s*\n")
MATH_OR_TIME = re.compile(
    r"\d\s*[-+*/x×÷^]\s*\d|cu[aá]nto (es|son|da)|calcula|multiplica|divide|suma|resta|ra[ií]z|porcentaje|"
    r"qu[eé] (hora|d[ií]a|fecha)|hoy",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calculator", type=int, default=8000)
    parser.add_argument("--datetime", type=int, default=3000)
    parser.add_argument("--documents", type=int, default=6000)
    parser.add_argument("--not-found", type=int, default=1000)
    parser.add_argument("--multi", type=int, default=1200)
    parser.add_argument("--no-tool", type=int, default=8000)
    parser.add_argument("--eval", type=int, default=100, help="held-out questions per tool for evaluate.py")
    parser.add_argument("--demo-docs", type=int, default=300)
    parser.add_argument("--pool-docs", type=int, default=2500)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def stable_fraction(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def tool_dialog(question: str, steps: list[tuple[dict, str]], answer: str) -> list[dict]:
    """A conversation: the question, one call + result per step, and the final answer."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for call, result in steps:
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": result},
        ]
    return [*messages, {"role": "assistant", "content": answer}]


def calc(expression: str) -> tuple[dict, str]:
    return {"name": "calculadora", "arguments": {"expresion": expression}}, calculate(expression)


# ---------------------------------------------------------------- calculadora


def number(rng: random.Random, decimals: bool = False) -> str:
    """A number as a Spanish speaker writes it: 7, 348, 12,5."""
    magnitude = rng.choice([10, 100, 100, 1000, 1000, 10000])
    value = rng.randint(2, magnitude)
    if decimals and rng.random() < 0.3:
        return f"{value // 10 or 1},{rng.randint(1, 9)}"
    return str(value)


ITEMS = [
    ("cajas", "libros"), ("bolsas", "naranjas"), ("filas", "sillas"), ("paquetes", "galletas"), ("equipos", "jugadores")
]  # fmt: skip
OPERATIONS = {
    "+": (
        ["¿Cuánto es {a} más {b}?", "Suma {a} y {b}.", "¿Cuánto da {a} + {b}?", "{a} + {b}", "¿Me sumas {a} y {b}?",
         "Calcula {a} más {b}.", "¿Cuál es la suma de {a} y {b}?"],
        ["{a} más {b} son {r}.", "La suma es {r}.", "{a} + {b} = {r}.", "Son {r}.", "El resultado es {r}."],
    ),
    "-": (
        ["¿Cuánto es {a} menos {b}?", "Resta {b} a {a}.", "{a} - {b}", "¿Cuánto da {a} menos {b}?",
         "Calcula {a} menos {b}.", "¿Cuál es la diferencia entre {a} y {b}?"],
        ["{a} menos {b} son {r}.", "La diferencia es {r}.", "{a} - {b} = {r}.", "El resultado es {r}."],
    ),
    "*": (
        ["¿Cuánto es {a} por {b}?", "Multiplica {a} por {b}.", "{a} x {b}", "¿Cuánto da {a} * {b}?",
         "Calcula {a} por {b}.", "¿Cuál es el producto de {a} y {b}?", "{a} × {b}"],
        ["{a} por {b} son {r}.", "El producto es {r}.", "{a} × {b} = {r}.", "Son {r}.", "El resultado es {r}."],
    ),
    "/": (
        ["¿Cuánto es {a} entre {b}?", "Divide {a} entre {b}.", "{a} / {b}", "¿Cuánto da {a} dividido entre {b}?",
         "Calcula {a} entre {b}."],
        ["{a} entre {b} son {r}.", "El cociente es {r}.", "{a} / {b} = {r}.", "El resultado es {r}."],
    ),
}  # fmt: skip


def calculator_dialog(rng: random.Random) -> list[dict]:
    kind = rng.random()
    if kind < 0.6:
        op = rng.choice(list(OPERATIONS))
        a, b = number(rng, decimals=True), number(rng, decimals=op in "+-")
        if op == "/":
            b = str(rng.randint(2, 50))
        questions, answers = OPERATIONS[op]
        expression = f"{a}{op}{b}"
        step = calc(expression)
        question = rng.choice(questions).format(a=a, b=b)
        answer = rng.choice(answers).format(a=a, b=b, r=step[1])
        return tool_dialog(question, [step], answer)
    if kind < 0.7:
        a, p = number(rng), rng.choice([5, 10, 12, 15, 20, 21, 25, 30, 50, 75])
        step = calc(f"{p}*{a}/100")
        question = rng.choice(
            ["¿Cuánto es el {p}% de {a}?", "Calcula el {p} por ciento de {a}.", "¿Qué es el {p}% de {a}?"]
        )
        answer = rng.choice(["El {p}% de {a} es {r}.", "Son {r}."])
        return tool_dialog(question.format(p=p, a=a), [step], answer.format(p=p, a=a, r=step[1]))
    if kind < 0.8:
        a = rng.randint(2, 99)
        if rng.random() < 0.5:
            step = calc(f"{a}^2")
            question = rng.choice(
                ["¿Cuánto es {a} al cuadrado?", "Calcula {a} elevado a 2.", "¿Cuál es el cuadrado de {a}?"]
            )
            answer = "{a} al cuadrado es {r}."
        else:
            a = rng.choice([a * a, rng.randint(2, 5000)])
            step = calc(f"raiz({a})")
            question = rng.choice(
                ["¿Cuál es la raíz cuadrada de {a}?", "Calcula la raíz de {a}.", "Raíz cuadrada de {a}"]
            )
            answer = "La raíz cuadrada de {a} es {r}."
        return tool_dialog(question.format(a=a), [step], answer.format(a=a, r=step[1]))
    if kind < 0.9:
        a, b, c = (rng.randint(2, 200) for _ in range(3))
        step = calc(f"({a}+{b})*{c}")
        question = rng.choice(
            ["¿Cuánto es ({a} + {b}) por {c}?", "Calcula ({a}+{b})*{c}.", "Suma {a} y {b} y multiplica por {c}."]
        )
        return tool_dialog(question.format(a=a, b=b, c=c), [step], f"El resultado es {step[1]}.")
    # Word problems: the model has to turn words into an expression.
    a, b = rng.randint(2, 60), rng.randint(2, 60)
    container, thing = rng.choice(ITEMS)
    problem = rng.choice(["mul", "sub", "div"])
    if problem == "mul":
        step = calc(f"{a}*{b}")
        question = f"Tengo {a} {container} con {b} {thing} cada una. ¿Cuántos {thing} hay en total?"
        answer = f"Hay {step[1]} {thing} en total."
    elif problem == "sub":
        a, b = max(a, b) * 10, min(a, b) * 3
        step = calc(f"{a}-{b}")
        question = f"Tenía {a} euros y gasté {b}. ¿Cuánto dinero me queda?"
        answer = f"Te quedan {step[1]} euros."
    else:
        total = a * b
        step = calc(f"{total}/{b}")
        question = f"Reparto {total} {thing} entre {b} personas a partes iguales. ¿Cuántos le tocan a cada una?"
        answer = f"A cada persona le tocan {step[1]} {thing}."
    return tool_dialog(question, [step], answer)


# ---------------------------------------------------------------- fecha_hora

DATE_QUESTIONS = {
    "fecha": (["¿Qué día es hoy?", "¿Qué fecha es hoy?", "Dime la fecha de hoy.", "¿A qué fecha estamos?",
               "¿Me dices qué día es hoy?"], "Hoy es {weekday}, {day} de {month} de {year}."),
    "hora": (["¿Qué hora es?", "Dime la hora.", "¿Me dices qué hora es?", "¿Qué hora tienes?"], "Son las {time}."),
    "año": (["¿En qué año estamos?", "¿Qué año es?"], "Estamos en {year}."),
    "semana": (["¿Qué día de la semana es hoy?", "¿Hoy qué día de la semana es?"], "Hoy es {weekday}."),
    "mes": (["¿En qué mes estamos?", "¿Qué mes es?"], "Estamos en {month}."),
}  # fmt: skip


def random_moment(rng: random.Random) -> datetime:
    return datetime(2024, 1, 1) + timedelta(minutes=rng.randint(0, 7 * 365 * 24 * 60))


def fields(moment: datetime) -> dict:
    return {
        "weekday": DAYS[moment.weekday()],
        "day": moment.day,
        "month": MONTHS[moment.month - 1],
        "year": moment.year,
        "time": f"{moment:%H:%M}",
    }


def datetime_dialog(rng: random.Random) -> list[dict]:
    moment = random_moment(rng)
    questions, answer = DATE_QUESTIONS[rng.choice(list(DATE_QUESTIONS))]
    step = ({"name": "fecha_hora", "arguments": {}}, format_datetime(moment))
    return tool_dialog(rng.choice(questions), [step], answer.format(**fields(moment)))


# ---------------------------------------------------------------- documentos

DOC_QUESTIONS = [
    "¿Qué dicen mis documentos sobre {t}?",
    "Busca en mis documentos información sobre {t}.",
    "¿Qué sabes de {t}? Mira en mis documentos.",
    "Según mis documentos, ¿qué es {t}?",
    "Consulta mis documentos: {t}",
    "¿Tengo algo en mis documentos sobre {t}?",
    "Mira en mis archivos qué pone de {t}.",
    "Resúmeme lo que dicen mis documentos de {t}.",
]
DOC_ANSWERS = ["Según tus documentos, {s}", "En tus documentos pone: {s}", "Tus documentos dicen que {s}"]
LOWER_START = {"El", "La", "Los", "Las", "Un", "Una", "En", "Es", "Se", "Fue", "Este", "Esta"}


def first_sentences(text: str, min_chars: int = 60) -> str | None:
    """The first sentence of a passage (two if the first is very short), or None if the passage
    was cut before the sentence ended: the answer must only say what the result says."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", text)
    picked = sentences[0]
    if len(picked) < min_chars and len(sentences) > 1:
        picked += " " + sentences[1]
    return picked if picked.endswith((".", "!", "?")) else None


def lower_start(sentence: str) -> str:
    word = sentence.split(" ", 1)[0]
    return sentence[0].lower() + sentence[1:] if word in LOWER_START else sentence


def documents_dialog(rng: random.Random, title: str, pool: Path) -> list[dict] | None:
    result = search_documents(title, pool)
    match = re.match(r"\[(.+?)\] (.*)", result)
    if not match or match.group(1) != title:
        return None  # the search did not put this article first: the answer would not be grounded
    sentence = first_sentences(match.group(2))  # the first hit (one line); None if it was cut short
    if sentence is None:
        return None
    template = rng.choice(DOC_ANSWERS)
    # "Según tus documentos, la ciudad…" but "En tus documentos pone: La ciudad…"
    answer = template.format(s=sentence if template.endswith(": {s}") else lower_start(sentence))
    call = {"name": "buscar_documentos", "arguments": {"consulta": title}}
    return tool_dialog(rng.choice(DOC_QUESTIONS).format(t=title), [(call, result)], answer)


def not_found_dialog(rng: random.Random, title: str, pool: Path) -> list[dict] | None:
    result = search_documents(title, pool)
    if not result.startswith("No se encontró"):
        return None
    call = {"name": "buscar_documentos", "arguments": {"consulta": title}}
    answer = rng.choice(
        [
            "No he encontrado nada sobre {t} en tus documentos.",
            "Tus documentos no dicen nada sobre {t}.",
            "No hay información sobre {t} en tus documentos.",
        ]
    ).format(t=title)
    return tool_dialog(rng.choice(DOC_QUESTIONS).format(t=title), [(call, result)], answer)


# ---------------------------------------------------------------- varias llamadas


def multi_dialog(rng: random.Random) -> list[dict]:
    if rng.random() < 0.5:
        moment = random_moment(rng)
        now = ({"name": "fecha_hora", "arguments": {}}, format_datetime(moment))
        if rng.random() < 0.5:
            target = moment.year + rng.randint(1, 80)
            step = calc(f"{target}-{moment.year}")
            question = rng.choice(["¿Cuántos años faltan para {y}?", "¿Cuánto falta para el año {y}?"]).format(y=target)
            return tool_dialog(question, [now, step], f"Faltan {step[1]} años para {target}.")
        past = moment.year - rng.randint(1, 600)
        step = calc(f"{moment.year}-{past}")
        question = rng.choice(["¿Cuántos años han pasado desde {y}?", "¿Hace cuántos años fue {y}?"]).format(y=past)
        return tool_dialog(question, [now, step], f"Han pasado {step[1]} años desde {past}.")
    a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 500)
    first = calc(f"{a}*{b}")
    second = calc(f"{first[1]}+{c}")
    question = rng.choice(
        ["Multiplica {a} por {b} y al resultado súmale {c}.", "Calcula {a} por {b}, y luego suma {c}."]
    ).format(a=a, b=b, c=c)
    return tool_dialog(question, [first, second], f"{a} por {b} son {first[1]}, y sumando {c} queda {second[1]}.")


# ---------------------------------------------------------------- sin herramienta


def no_tool_dialogs(rng: random.Random, count: int, split: str = "train") -> list[list[dict]]:
    """Phase 5 conversations that need no tool, given the tools' system prompt."""
    path = DATA_DIR / "sft" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run phases/5_sft/prepare.py first.")
    candidates = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            messages = [m for m in json.loads(line)["messages"] if m["role"] != "system"]
            if MATH_OR_TIME.search(messages[0]["content"]) or len(" ".join(m["content"] for m in messages)) > 2500:
                continue
            candidates.append(messages)
    rng.shuffle(candidates)
    return [[{"role": "system", "content": SYSTEM}, *messages] for messages in candidates[:count]]


# ---------------------------------------------------------------- documents on disk


def slug(title: str) -> str:
    text = re.sub(r"[^\w\s-]", "", title.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", "_", text).strip("_")[:60] or "documento"


def write_documents(demo: int, pool: int) -> tuple[list[str], list[str], list[str]]:
    """Wikipedia val articles as .md: demo docs, training pool, and titles of a third set (absent)."""
    if not WIKI_VAL.exists():
        raise SystemExit("Missing data/wiki_es/val.jsonl. Run phases/4_pretrain/control.py prepare first.")
    articles = []
    with open(WIKI_VAL, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            title, _, body = text.partition("\n\n")
            body = END_SECTIONS.split(body + "\n")[0].strip()  # drop "Referencias", "Enlaces externos"...
            if 1500 <= len(text) <= 8000 and 3 <= len(title) <= 40 and "(" not in title:
                articles.append((title.strip(), body.strip()))
            if len(articles) >= demo + pool * 2:
                break
    seen, unique = set(), []
    for title, body in articles:
        if title.lower() not in seen:
            seen.add(title.lower())
            unique.append((title, body))
    groups = {DEMO_DIR: unique[:demo], POOL_DIR: unique[demo : demo + pool]}
    for folder, items in groups.items():
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.md"):
            old.unlink()
        for title, body in items:
            (folder / f"{slug(title)}.md").write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    absent = [title for title, _ in unique[demo + pool :]]
    return [t for t, _ in groups[DEMO_DIR]], [t for t, _ in groups[POOL_DIR]], absent


# ---------------------------------------------------------------- evaluation set


def eval_items(rng: random.Random, n: int, demo_titles: list[str], absent: list[str]) -> list[dict]:
    """Held-out questions with what a right answer needs: the tool and, when checkable, the value."""
    items = []
    for _ in range(n):
        messages = calculator_dialog(rng)
        call = messages[2]["tool_calls"][0]
        items.append({"kind": "calculadora", "question": messages[1]["content"], "tool": "calculadora",
                      "expected": messages[3]["content"], "expression": call["arguments"]["expresion"]})  # fmt: skip
    for _ in range(n):
        questions, _ = DATE_QUESTIONS[rng.choice(list(DATE_QUESTIONS))]
        items.append({"kind": "fecha_hora", "question": rng.choice(questions), "tool": "fecha_hora"})
    for title in rng.sample(demo_titles, min(n, len(demo_titles))):
        items.append({"kind": "documentos", "question": rng.choice(DOC_QUESTIONS).format(t=title),
                      "tool": "buscar_documentos", "title": title})  # fmt: skip
    for title in rng.sample(absent, min(n // 2, len(absent))):
        items.append({"kind": "no_encontrado", "question": rng.choice(DOC_QUESTIONS).format(t=title),
                      "tool": "buscar_documentos", "title": title})  # fmt: skip
    for messages in no_tool_dialogs(random.Random(rng.random()), n, split="val"):
        items.append({"kind": "sin_herramienta", "question": messages[1]["content"], "tool": None})
    return items


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tokenizer = load_tokenizer()

    print("Writing documents from Wikipedia validation articles...", flush=True)
    demo_titles, pool_titles, absent = write_documents(args.demo_docs, args.pool_docs)
    print(f"  {len(demo_titles)} demo documents in {DEMO_DIR}, {len(pool_titles)} in {POOL_DIR}", flush=True)

    dialogs: list[tuple[str, list[dict]]] = []
    dialogs += [("calculadora", calculator_dialog(rng)) for _ in range(args.calculator)]
    dialogs += [("fecha_hora", datetime_dialog(rng)) for _ in range(args.datetime)]
    dialogs += [("varias", multi_dialog(rng)) for _ in range(args.multi)]
    print("Searching the document pool...", flush=True)
    for kind, count, make, titles in (
        ("documentos", args.documents, documents_dialog, pool_titles),
        ("no_encontrado", args.not_found, not_found_dialog, absent),
    ):
        made = 0
        for title in rng.sample(titles, len(titles)) * 3:
            if made >= count:
                break
            if (dialog := make(rng, title, POOL_DIR)) is not None:
                dialogs.append((kind, dialog))
                made += 1
        print(f"  {kind}: {made:,}", flush=True)
    dialogs += [("sin_herramienta", d) for d in no_tool_dialogs(rng, args.no_tool)]

    splits: dict[str, list] = {"train": [], "val": []}
    stats = {"tokenizer": DEFAULT_TOKENIZER, "dropped": Counter()}
    seen = set()
    for kind, messages in dialogs:
        key = json.dumps(messages, ensure_ascii=False)
        if key in seen:
            stats["dropped"]["duplicate"] += 1
            continue
        seen.add(key)
        tokens, targets = encode_chat_for_training(tokenizer, messages)
        if len(tokens) > args.max_tokens:
            stats["dropped"]["too long"] += 1
            continue
        split = "val" if stable_fraction(messages[1]["content"]) < 0.05 else "train"
        splits[split].append((kind, messages, tokens, targets))

    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        rng.shuffle(items)
        with open(AGENT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for kind, messages, _, _ in items:
                f.write(json.dumps({"source": kind, "messages": messages}, ensure_ascii=False) + "\n")
        counts = save_conversations(
            AGENT_DIR / f"{split}.npz",
            [(tokens, targets) for _, _, tokens, targets in items],
            [kind for kind, *_ in items],
            DEFAULT_TOKENIZER,
        )
        counts["by_source"] = dict(Counter(kind for kind, *_ in items))
        stats[split] = counts
        print(f"{split}: {counts}", flush=True)

    evaluation = eval_items(random.Random(args.seed + 1), args.eval, demo_titles, absent)
    with open(AGENT_DIR / "eval.jsonl", "w", encoding="utf-8") as f:
        for item in evaluation:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    stats["eval"] = dict(Counter(item["kind"] for item in evaluation))
    stats["dropped"] = dict(stats["dropped"])
    (AGENT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"eval: {stats['eval']}\ndropped: {stats['dropped']}")


if __name__ == "__main__":
    main()
