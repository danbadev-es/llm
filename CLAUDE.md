# Notes for Claude Code sessions in this repo

Read README.md and ARCHITECTURE.md first. Answer the user in Spanish; code, comments, identifiers and commit messages in English.

## Project status

The project is **finished** (phases 0–8; the user stopped it: "es suficiente para el objetivo educativo"). No training run is pending. Phase 8's GRPO (`control.py grpo`) and `evaluate.py` were never run on the final model: they are left as exercises, so only run them if the user asks. The best chat model is `checkpoints/7_dpo/leon_0.1_dpo/best.pt` (stage `agent`, supports tools).

## Training runs: what the user's words mean

Every long run (phases 4–8) has a `control.py` in its phase folder, built on `leon.run` (`Control` in phases 5–8; phase 4 has its own commands on the same helpers). Run them from the repo root with `.venv\Scripts\python.exe`. The words act on the phase the user names; if they name none, ask, since every run has finished.

| The user says | Run | Then |
|---|---|---|
| "prepara los datos" | phase 4: `control.py prepare` (CPU + network, 20–40 min, in the background); phase 5: `prepare.py` (CPU + network); phases 6 and 8: `prepare.py` (CPU); phase 7: `control.py prepare` (GPU, ~35 min: needs the user's go-ahead like any GPU run; its status is `status --run prepare`) | report what it wrote (tokens per split, conversations, pairs or problems) |
| "arranca" / "empieza" / "lanza" | `control.py start` (phase 8: `start` for `leon_0.1_steps`, `start --run leon_0.1_direct` for the control, `grpo` for GRPO) | report the pid and the first status |
| "pausa" / "para" | `control.py pause` | report the step it saved at |
| "continúa" / "reanuda" | `control.py resume` (phase 8 GRPO: `grpo` again) | report the step it resumed from |
| "¿cómo va?" / "estado" | `control.py status` | summarize progress, loss, ETA and the latest sample or answers |
| "muéstrame el progreso" / "abre el panel" | `control.py watch` in its own window (`leon.run.open_watch_window`) | say that closing it does not stop training |
| "extiende el contexto" / "arranca 2048" | phase 4 `control.py extend` (only once `leon_0.1` has finished) | report the pid and, at the first eval, the loss on both halves of the context |
| "evalúa" | phases 6–8: `evaluate.py` (uses the GPU if free: ask first) | report its table |

Default runs: phase 4 `leon_0.1_ctx2048` once `extend` has run (pass `--run leon_0.1` for the base run), phase 5 `leon_0.1_chat`, phase 6 `leon_0.1_agent`, phase 7 `leon_0.1_dpo`, phase 8 `leon_0.1_steps` (also `leon_0.1_direct` and `leon_0.1_grpo`).

- Never use `start --fresh` (it discards hours of progress) unless the user explicitly asks to start over.
- Do not launch long GPU runs on your own initiative; the user decides when the laptop is free.
- The trainer is a windowless background process: it survives the end of the chat session. Check it with `status`, never by killing it. `start`/`resume` open a `watch` dashboard window; closing it is harmless.
- A new long run follows the same pattern: the trainer writes `status.json` through `leon.run.RunFiles`, and its `control.py` is a `leon.run.Control`, so every phase has the same commands and window (see CONTRIBUTING.md).

## Quality checks before every commit

`ruff check .`, `ruff format --check .`, `pytest`, `python phases/1_gpt/test_model.py`, and `npm run build` in `apps/web` if the web changed.
