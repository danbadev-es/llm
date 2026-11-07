"""Phase 6: measure tool calling on held-out questions (data/agent/eval.jsonl).

Runs the real agent loop (leon.agent, the same as the chat app) on each question and checks:

    JSON válido      the calls the model wrote parse as {"name", "arguments"}
    herramienta      it called the expected tool (or none, for questions that need none)
    respuesta        the answer uses the result: the calculator's value, part of the date,
                     the document's text, or says that nothing was found

Questions about documents use data/docs/ejemplo/, which the training dialogs never searched.
Targets (docs/fundamentos.md): valid JSON > 95 %, right tool > 80 %.

    python phases/6_agent/evaluate.py                      # the phase 6 model
    python phases/6_agent/evaluate.py --checkpoint checkpoints/5_sft/leon_0.1_chat/best.pt
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from leon.agent import run_agent
from leon.chat import encode_chat
from leon.data import load_tokenizer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.tools import calculate, system_prompt, words

DEFAULT = CHECKPOINT_DIR / "6_agent" / "leon_0.1_agent" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, reproducible")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="questions per kind (0 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load(path: Path, device: str) -> tuple[Leon, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Leon(LeonConfig(**checkpoint["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, load_tokenizer(checkpoint.get("tokenizer", "v1"))


def answer_is_right(item: dict, answer: str, results: list[str]) -> bool:
    """Whether the final answer uses what the tools returned (or says nothing was found)."""
    text = answer.lower()
    if item["kind"] == "calculadora":
        return calculate(item["expression"]).lower() in text
    if item["kind"] == "fecha_hora":
        parts = re.findall(r"\d+:\d+|\w+", results[-1].lower()) if results else []
        return any(part in text for part in parts if len(part) > 2)  # weekday, month, year or time
    if item["kind"] == "documentos":
        found, said = set(words(" ".join(results))), set(words(answer))
        return bool(said) and len(said & found) / len(said) >= 0.6
    if item["kind"] == "no_encontrado":
        return bool(re.search(r"\bno\b", text))
    return True  # sin_herramienta: only the tool choice is measured


def main() -> None:
    args = parse_args()
    use_bf16 = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    model, tokenizer = load(args.checkpoint, args.device)
    block_size = model.config.block_size
    system = system_prompt()

    @torch.no_grad()
    def generate(ids, max_new, stop_ids):
        idx = torch.tensor([ids[-block_size:]], device=args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            for token in model.stream(idx, max_new, temperature=args.temperature, top_k=40, stop_ids=stop_ids):
                yield token.item()

    items = [json.loads(line) for line in open(DATA_DIR / "agent" / "eval.jsonl", encoding="utf-8")]
    counts = defaultdict(lambda: defaultdict(int))
    examples = []
    for item in items:
        kind = counts[item["kind"]]
        if args.limit and kind["n"] >= args.limit:
            continue
        prompt = encode_chat(
            tokenizer, [{"role": "system", "content": system}, {"role": "user", "content": item["question"]}]
        )
        calls, results, text = [], [], ""
        for event in run_agent(generate, tokenizer, prompt, args.max_new_tokens):
            if "tool_call" in event:
                calls.append(event["tool_call"])
            elif "tool_result" in event:
                results.append(event["tool_result"])
            elif "text" in event:
                text += event["text"]
        valid = [c for c in calls if "error" not in c]
        first = calls[0].get("name") if calls else None
        kind["n"] += 1
        kind["calls"] += len(calls)
        kind["valid_calls"] += len(valid)
        kind["right_tool"] += first == item["tool"]
        kind["right_answer"] += first == item["tool"] and answer_is_right(item, text, results)
        if len(examples) < 40 and kind["n"] <= 4:
            examples.append({"kind": item["kind"], "question": item["question"], "calls": calls, "answer": text})

    print(f"Model: {args.checkpoint}\n")
    print("| Tipo | Preguntas | JSON válido | Herramienta correcta | Respuesta correcta |")
    print("|---|---|---|---|---|")
    total = defaultdict(int)
    for name, kind in counts.items():
        for key, value in kind.items():
            total[key] += value
        json_rate = f"{100 * kind['valid_calls'] / kind['calls']:.0f} %" if kind["calls"] else "—"
        print(
            f"| {name} | {kind['n']} | {json_rate} | {100 * kind['right_tool'] / kind['n']:.0f} % "
            f"| {100 * kind['right_answer'] / kind['n']:.0f} % |"
        )
    json_total = f"{100 * total['valid_calls'] / total['calls']:.0f} %" if total["calls"] else "—"
    print(
        f"| **total** | {total['n']} | {json_total} | {100 * total['right_tool'] / total['n']:.0f} % "
        f"| {100 * total['right_answer'] / total['n']:.0f} % |"
    )
    out = args.checkpoint.parent / "eval.json"
    report = {"counts": {k: dict(v) for k, v in counts.items()}, "examples": examples}
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nExamples and counts: {out}")


if __name__ == "__main__":
    main()
