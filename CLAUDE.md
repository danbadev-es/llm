# Notes for Claude Code sessions in this repo

Read README.md and ARCHITECTURE.md first. Answer the user in Spanish; code, comments, identifiers and commit messages in English.

## Phase 4 training: what the user's words mean

The pretraining run is controlled with `phases/4_pretrain/control.py` (run from the repo root with `.venv\Scripts\python.exe`):

| The user says | Run | Then |
|---|---|---|
| "prepara los datos" (phase 4) | `control.py prepare` (CPU + network, 20–40 min: run it in the background) | report tokens per split from `data/wiki_es/*.json` |
| "arranca" / "empieza" | `control.py start` | report the pid and the first status |
| "pausa" / "para" | `control.py pause` | report the step it saved at |
| "continúa" / "reanuda" | `control.py resume` | report the step it resumed from |
| "¿cómo va?" / "estado" | `control.py status` | summarize progress, loss, ETA and the latest sample |
| "muéstrame el progreso" / "abre el panel" | `control.py watch` in its own window (`open_watch_window`) | say that closing it does not stop training |

- Never use `start --fresh` (it discards hours of progress) unless the user explicitly asks to start over.
- Do not launch long GPU runs on your own initiative; the user decides when the laptop is free.
- The trainer is a windowless background process: it survives the end of the chat session. Check it with `status`, never by killing it. `start`/`resume` open a `watch` dashboard window; closing it is harmless.

## Quality checks before every commit

`ruff check .`, `ruff format --check .`, `pytest`, `python phases/1_gpt/test_model.py`, and `npm run build` in `apps/web` if the web changed.
