"""Phase 8: reinforcement learning with a verifiable reward (GRPO, leon/grpo.py).

Starts from the step-by-step model (train.py's leon_0.1_steps) and, at every step:

    1. takes --problems problems (data/reasoning/problems.jsonl, levels 2–5, with their answers)
    2. samples --group answers to each one (one batch per problem: the prompt repeated)
    3. rewards each answer: 1 if its "Respuesta: N" is right, 0.1 if only the format is
    4. advantage = how much better than its own group; pushes the good answers' tokens up and
       the bad ones' down, with a KL penalty towards the starting model (the frozen reference)

Nobody shows the model how to think here: it only learns which of its own attempts worked.
Every --eval-interval steps, greedy accuracy on held-out problems (data/reasoning/eval.jsonl).
Usually run through phases/8_reasoning/control.py grpo (pause / status / watch with --run leon_0.1_grpo).
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, replace

import torch

from leon.chat import encode_chat
from leon.data import load_tokenizer
from leon.dpo import batch
from leon.grpo import extract_answer, group_advantages, grpo_loss, reward, token_logprobs
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import RunFiles, keep_system_awake
from leon.training import load_checkpoint, lr_at, save_checkpoint

CHECKPOINT_ROOT = CHECKPOINT_DIR / "8_reasoning"
REASONING_DIR = DATA_DIR / "reasoning"
BASE = CHECKPOINT_ROOT / "leon_0.1_steps" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--run", default="leon_0.1_grpo")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--problems", type=int, default=8, help="problems per step")
    parser.add_argument("--group", type=int, default=8, help="answers sampled per problem")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="varied attempts: a group of equal ones teaches nothing"
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty towards the reference")
    parser.add_argument("--micro-batch", type=int, default=16, help="sequences per forward pass")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-problems", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(path, device) -> tuple[Leon, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Leon(replace(LeonConfig(**checkpoint["config"]), dropout=0.0)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


@torch.no_grad()
def sample(model, tokenizer, question: str, count: int, args, autocast) -> tuple[list[int], list[list[int]]]:
    """The prompt's ids and `count` sampled answers (token ids, ending with <|end|> if they ended)."""
    prompt = encode_chat(tokenizer, [{"role": "user", "content": question}])
    idx = torch.tensor([prompt] * count, device=args.device)
    stop = set(tokenizer.special_tokens.values())
    end = tokenizer.special_tokens["<|end|>"]
    answers: list[list[int]] = [[] for _ in range(count)]
    finished = [False] * count
    temperature = args.temperature if count > 1 else 0.0  # a single answer is the greedy one (evaluation)
    model.eval()
    with autocast():
        for step in model.stream(idx, args.max_new_tokens, temperature=temperature, top_k=50, stop_ids=stop):
            for row, token in enumerate(step.view(-1).tolist()):
                if finished[row]:
                    continue
                if token in stop:
                    finished[row] = True
                    if token == end:
                        answers[row].append(token)  # learned: knowing when to stop
                else:
                    answers[row].append(token)
    return prompt, answers


def accuracy(model, tokenizer, problems: list[dict], args, autocast) -> float:
    right = 0
    for item in problems:
        _, (answer,) = sample(model, tokenizer, item["question"], 1, args, autocast)
        right += extract_answer(tokenizer.decode(answer)) == item["answer"]
    return right / len(problems)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    run_dir = CHECKPOINT_ROOT / args.run
    policy, base = load_model(args.base, device)
    reference, _ = load_model(args.base, device)
    reference.eval().requires_grad_(False)
    tokenizer = load_tokenizer(base.get("tokenizer", "v1"))
    pad_id, stage = tokenizer.eot_id, base.get("stage", "chat")
    problems = [json.loads(line) for line in open(REASONING_DIR / "problems.jsonl", encoding="utf-8")]
    random.Random(args.seed).shuffle(problems)
    held_out = [json.loads(line) for line in open(REASONING_DIR / "eval.jsonl", encoding="utf-8")]
    held_out = random.Random(0).sample(held_out, min(args.eval_problems, len(held_out)))

    optimizer = policy.configure_optimizer(args.lr, 0.0, (0.9, 0.95), device)
    start_step, best_val = 0, float("inf")  # best_val: the error rate (1 − accuracy)
    if args.resume:
        checkpoint = load_checkpoint(run_dir / "last.pt", policy, optimizer, device)
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
        print(f"Resumed from step {start_step}", flush=True)
    else:
        print(f"Base model: {args.base} (step {base['step']})", flush=True)
    print(
        f"{len(problems):,} problems | {args.problems} x {args.group} answers per step | {args.steps} steps\n",
        flush=True,
    )
    settings = {**vars(args), "model": asdict(policy.config)}

    def save(path, step, with_optimizer):
        state = optimizer if with_optimizer else None
        save_checkpoint(
            path, policy, step, best_val, state, settings, stage=stage, tokenizer=base.get("tokenizer", "v1")
        )

    run = RunFiles(run_dir)
    run.dir.mkdir(parents=True, exist_ok=True)
    run.pid.write_text(str(os.getpid()))
    run.clear_pause()
    keep_system_awake()
    status = {"state": "running", "max_steps": args.steps}
    if args.resume:
        status.update({k: v for k, v in run.read_status().items() if k in ("metrics", "probes")})

    def report(step: int, **fields) -> None:
        status.update(fields)
        run.write_status(**status, step=step)

    report(start_step)
    step, window_start, window_steps = start_step, time.perf_counter(), 0
    try:
        for step in range(start_step, args.steps + 1):
            if (step % args.eval_interval == 0 or step == args.steps) and not (args.resume and step == start_step):
                score = accuracy(policy, tokenizer, held_out, args, autocast)
                example = held_out[0]
                _, (answer,) = sample(policy, tokenizer, example["question"], 1, args, autocast)
                print(
                    f"step {step:4d}/{args.steps} | val accuracy {score:.3f} (greedy, {len(held_out)} problems)",
                    flush=True,
                )
                report(
                    step,
                    metrics={**status.get("metrics", {}), "val accuracy": score},
                    probes=[[example["question"], tokenizer.decode(answer).replace("<|end|>", "")[:300]]],
                )
                window_start, window_steps = time.perf_counter(), 0
                if 1 - score < best_val:
                    best_val = 1 - score
                    save(run_dir / "best.pt", step, with_optimizer=False)
                save(run_dir / "last.pt", step, with_optimizer=True)
            if step == args.steps:
                break

            # 1–3: sample a group of answers per problem and reward them.
            chunk = [problems[(step * args.problems + i) % len(problems)] for i in range(args.problems)]
            sequences, rewards, lengths = [], [], []
            for item in chunk:
                prompt, answers = sample(policy, tokenizer, item["question"], args.group, args, autocast)
                group = [reward(tokenizer.decode(answer), item["answer"]) for answer in answers]
                rewards.append(group)
                for answer in answers:
                    sequences.append((prompt + answer, [False] * len(prompt) + [True] * len(answer)))
                    lengths.append(len(answer))
            rewards_t = torch.tensor(rewards)
            advantages = group_advantages(rewards_t).view(-1).to(device)

            # 4: policy-gradient update with the KL penalty, in micro-batches.
            policy.train()
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.lr, args.lr, args.warmup_steps, args.steps)
            kl_total = 0.0
            for start in range(0, len(sequences), args.micro_batch):
                part = sequences[start : start + args.micro_batch]
                x, y = batch(part, pad_id, device)
                with autocast():
                    logprobs, mask = token_logprobs(policy, x, y)
                    with torch.no_grad():
                        reference_logprobs, _ = token_logprobs(reference, x, y)
                loss, metrics = grpo_loss(
                    logprobs, reference_logprobs, mask, advantages[start : start + len(part)], args.beta
                )
                (loss * len(part) / len(sequences)).backward()
                kl_total += metrics["kl"] * len(part) / len(sequences)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_steps += 1

            done = step + 1
            seconds = (time.perf_counter() - window_start) / window_steps
            learning = (rewards_t.std(dim=1) > 0).float().mean().item()
            report(
                done,
                train_loss=None,
                eta_hours=(args.steps - done) * seconds / 3600,
                metrics={
                    "reward": rewards_t.mean().item(),
                    "right": (rewards_t == 1).float().mean().item(),
                    "groups that teach": learning,
                    "length": sum(lengths) / len(lengths),
                    "kl": kl_total,
                    **{k: v for k, v in status.get("metrics", {}).items() if k == "val accuracy"},
                },
            )
            if done % 10 == 0:
                shown = status["metrics"]
                print(
                    f"step {done:4d} | reward {shown['reward']:.3f} | right {shown['right']:.3f} "
                    f"| groups that teach {learning:.2f} | length {shown['length']:.0f} | kl {kl_total:.4f}",
                    flush=True,
                )
            if run.pause_requested():
                save(run_dir / "last.pt", done, with_optimizer=True)
                run.clear_pause()
                report(done, state="paused")
                print(f"\nPaused at step {done}. Continue with: control.py grpo", flush=True)
                return
    except KeyboardInterrupt:
        save(run_dir / "last.pt", step, with_optimizer=True)
        report(step, state="paused")
        return

    report(args.steps, state="finished", eta_hours=0)
    print(f"\nBest val accuracy: {1 - best_val:.3f}. Model: {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
