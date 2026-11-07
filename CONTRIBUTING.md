# Guía de desarrollo

Cómo trabajar en el repositorio: preparar el entorno, el flujo de cada cambio y las convenciones. La organización del código está en [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Entorno

Windows, Python 3.11 o posterior, una GPU NVIDIA con CUDA 12.8+ para entrenar, y Node.js 20 o posterior para la web.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[all]"
cd apps/web; npm install; cd ../..
```

Los datos y los checkpoints no están en el repositorio. Para reconstruirlos, sigue en orden la sección "Ejecución" del README de cada fase: [`0_bigram`](phases/0_bigram/README.md) y [`2_tokenizer`](phases/2_tokenizer/README.md) para los datos base, y después el `prepare.py` y el `control.py` de las fases 4 a 8 (cada una parte del modelo de la anterior).

## Antes de cada commit

```powershell
ruff check .            # lint
ruff format --check .   # formato (sin --check, lo arregla)
pytest                  # 78 tests, CPU; no necesitan data/ ni checkpoints/
python phases/1_gpt/test_model.py
cd apps/web; npm run build; cd ../..    # si cambió la web
```

Si cambia un notebook (fases 0–3) o el código que muestra, vuelve a ejecutarlo completo para que las salidas guardadas sigan siendo reales.

## Convenciones

- **Idiomas**: la documentación (README, ARCHITECTURE, notebooks) se escribe en español. El código, los comentarios, los identificadores y los mensajes de commit, en inglés.
- **Claridad antes que optimización**: el código es material de estudio. Solo se optimiza cuando una medición lo justifica, y la medición se documenta (ver `phases/3_modern/README.md`).
- **Abstracciones proporcionales**: nada de capas genéricas "por si acaso". El código pasa a `leon/` cuando lo usan dos sitios, no antes.
- **Cada afirmación con su medición**: los resultados de los README salen de ejecuciones reales, con el hardware y las versiones indicados.
- **Métricas comunes** en todas las fases, para poder comparar:
  - val loss y bits por byte (bits por carácter en las fases 0 y 1);
  - tok/s;
  - una muestra con el prompt fijo `En un lugar de la Mancha`;
  - el mismo split 90/10.
- **Entrenamientos reanudables**: todo script de entrenamiento guarda `last.pt` con `leon.training.save_checkpoint` y acepta `--resume`.
- **Entrenamientos largos en segundo plano**: el trainer escribe `status.json` con `leon.run.RunFiles` y respeta el fichero `PAUSE`; su `control.py` es un `leon.run.Control`, así que todas las fases tienen los mismos comandos y el mismo panel.

## Añadir una fase

1. Crea `phases/N_nombre/` con sus scripts y un `README.md`. Un `walkthrough.ipynb` es opcional (lo tienen las fases 0–3).
2. Si entrena durante más de unos minutos, sigue el patrón de las fases 5–8:
   - `prepare.py` deja los datos en `data/<nombre>/` y anota en su `stats.json` con qué tokenizer se codificaron;
   - `train.py` escribe `status.json` (`state`, `step`, `max_steps`, pérdidas, `eta_hours`, `probes`...) con `leon.run.RunFiles`, se pausa limpiamente con `PAUSE` y guarda con `save_checkpoint(..., stage=...)`. Si es un ajuste supervisado sobre conversaciones, usa `leon.sft.train_sft`, como las fases 5, 6 y 8;
   - `control.py` declara un `leon.run.Control(title, runs_dir, script, control, default_run, check=...)`, con comandos extra si hacen falta (`prepare` en la fase 7, `grpo` en la 8);
   - si mide algo más que la loss, un `evaluate.py` con preguntas que no estén en el entrenamiento.
3. El README sigue la estructura de las fases anteriores:
   - objetivo;
   - archivos;
   - conceptos nuevos;
   - ejecución;
   - paso a paso;
   - qué observar;
   - ejercicios;
   - resultados.
4. Los checkpoints van a `checkpoints/N_nombre/<run>/`. El chat los detecta solo si la fase es la 3 o posterior, y elige modo y herramientas según su `stage` (`base`, `chat` o `agent`).
5. El código que vayan a reutilizar otras fases o las apps se mueve a `leon/`, con tests en `tests/` que corran en CPU sin `data/` ni `checkpoints/` (como `tests/test_sft.py` o `tests/test_rl.py`). Una fase nunca importa de otra (`tests/test_architecture.py`).
6. Añade la fase a la tabla de estado y a la hoja de ruta del [`README.md`](README.md), a [`ARCHITECTURE.md`](ARCHITECTURE.md) (módulos nuevos y "Cómo encaja cada fase"), a [`CLAUDE.md`](CLAUDE.md) si tiene `control.py`, y una entrada en [`CHANGELOG.md`](CHANGELOG.md).

## Commits

- Mensajes en inglés y en imperativo: `Add phase 4 pretraining loop`, `Fix off-by-one in get_batch`.
- Un commit por cambio lógico, con la suite de calidad en verde.
- `data/`, `checkpoints/` y cualquier archivo de pesos están en `.gitignore`. Nunca deben entrar en el historial.
