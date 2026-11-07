# Notes for Claude Code sessions in this repo

Read README.md and ARCHITECTURE.md first. Answer the user in Spanish; code, comments, identifiers and commit messages in English.

## Training runs: what the user's words mean

Every long training run has a `control.py` in its phase folder (run from the repo root with `.venv\Scripts\python.exe`). The words act on **the phase in progress**: the lowest phase whose run has not finished (phase 5, then 6: `phases/6_agent/control.py`, which needs `phases/6_agent/prepare.py` first, CPU only, and the phase 5 model); an earlier phase if the user names it. After phase 6 trains, run `phases/6_agent/evaluate.py` and report its table. Phase 7 (`phases/7_dpo/control.py`) has two GPU steps: `prepare` (sampling the pairs, ~30 min; its status is `status --run prepare`) and then `start`; "prepara los datos" in phase 7 means `prepare`, and it needs the user's go-ahead like any GPU run. Phase 8 (`phases/8_reasoning/control.py`): `prepare.py` first (CPU), then `start` (steps), `start --run leon_0.1_direct` (control) and `grpo`; report `evaluate.py`'s table at the end.

| The user says | Run | Then |
|---|---|---|
| "prepara los datos" (phase 4) | `control.py prepare` (CPU + network, 20–40 min: run it in the background) | report tokens per split from `data/wiki_es/*.json` |
| "arranca" / "empieza" / "lanza" | `control.py start` | report the pid and the first status |
| "pausa" / "para" | `control.py pause` | report the step it saved at |
| "continúa" / "reanuda" | `control.py resume` | report the step it resumed from |
| "¿cómo va?" / "estado" | `control.py status` | summarize progress, loss, ETA and the latest sample or answers |
| "muéstrame el progreso" / "abre el panel" | `control.py watch` in its own window (`leon.run.open_watch_window`) | say that closing it does not stop training |
| "extiende el contexto" / "arranca 2048" | phase 4 `control.py extend` (only once `leon_0.1` has finished) | report the pid and, at the first eval, the loss on both halves of the context |

Phase 4: `pause`, `resume`, `status` and `watch` act on the latest started stage (`leon_0.1_ctx2048` once `extend` has run); pass `--run leon_0.1` for the base run. Phase 5's run is `leon_0.1_chat`.

- Never use `start --fresh` (it discards hours of progress) unless the user explicitly asks to start over.
- Do not launch long GPU runs on your own initiative; the user decides when the laptop is free.
- The trainer is a windowless background process: it survives the end of the chat session. Check it with `status`, never by killing it. `start`/`resume` open a `watch` dashboard window; closing it is harmless.
- New phases with long runs (6–8) follow the same pattern: the trainer writes `status.json` through `leon.run.RunFiles`, and its `control.py` reuses `leon.run` (`launch`, `wait_for_pause`, `dashboard`, `watch`), so every phase has the same window.

## Quality checks before every commit

`ruff check .`, `ruff format --check .`, `pytest`, `python phases/1_gpt/test_model.py`, and `npm run build` in `apps/web` if the web changed.
