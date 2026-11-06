# llm: Leon 0.1

Proyecto educativo: construir desde cero **Leon 0.1**, un modelo de lenguaje conversacional en español que, al final, tendrá capacidades agénticas (tool calling).

Todo está implementado a mano: tokenizer, transformer, entrenamiento y chat, sin Hugging Face Transformers. Se entrena en local en una RTX 5060 Laptop (8 GB).

Cada fase vive en su propia carpeta, produce algo ejecutable y medible, y documenta en su `README.md` qué se aprendió y qué métricas obtuvo. Cada fase incluye además un `walkthrough.ipynb` que reconstruye sus piezas paso a paso con ejemplos diminutos.

## Estado

| Fase | Estado | Resultado clave |
|---|---|---|
| 0: bigrama | ✅ | 3,327 bits/char |
| 1: GPT mínimo | ✅ | 1,627 bits/char · SDPA ×1,76 |
| 2: tokenizer BPE | ✅ | 16.384 tokens · 4,19 bytes/token en Wikipedia |
| 3: arquitectura moderna | ✅ | 1,175 bits/byte (32M, 31 min) · tamaño de Leon 0.1 fijado |
| 4: preentrenamiento | ⏳ listo para arrancar | 56M, contexto 1024, ~2B tokens, ~13–14 h · `control.py start / pause / resume / status` |
| 5–7: SFT, agente, DPO | — | |

## Inicio rápido

Con el entorno instalado (ver [Entorno](#entorno-windows)) y los datos y checkpoints generados por las fases:

```powershell
pytest                                            # ¿todo en orden?
python phases/3_modern/sample.py --prompt "La ciudad de Madrid"
python apps/api/app.py                            # chat en http://127.0.0.1:8000
```

## Documentación

| Documento | Para qué |
|---|---|
| [`docs/fundamentos.md`](docs/fundamentos.md) | Conceptos desde cero, objetivo y criterios de éxito de Leon 0.1 |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Componentes, flujo de datos, formatos compartidos, decisiones de diseño y reglas de dependencias |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Entorno, flujo de trabajo, convenciones y cómo añadir una fase |
| [`CHANGELOG.md`](CHANGELOG.md) | Hitos y resultados, fase a fase |
| [`CLAUDE.md`](CLAUDE.md) | Instrucciones para sesiones de Claude Code: qué comando corresponde a "arranca", "pausa", "continúa"... |
| `phases/N_*/README.md` | Cada fase: conceptos, ejecución, ejercicios y resultados medidos |
| [`apps/README.md`](apps/README.md) | Chat local: API, web, modos y seguridad |

## Estructura del repositorio

```
llm/
├── pyproject.toml      # makes the `leon` package installable: pip install -e .
├── leon/               # the product: reusable, tested code shared by phases and apps
│   ├── tokenizer.py    #   byte-level BPE                         (built in phase 2)
│   ├── model.py        #   Leon: RoPE, RMSNorm, SwiGLU, GQA, KV cache (built in phase 3)
│   ├── data.py         #   token files with np.memmap             (built in phase 3)
│   ├── chat.py         #   chat template                          (phase 5 and apps/api)
│   ├── training.py     #   LR schedule, eval, bits/byte, atomic checkpoints with RNG state
│   ├── run.py          #   control files of a background run: pause flag, status, pid
│   └── paths.py        #   where data/ and checkpoints/ live
├── phases/             # the course: one chapter per phase (scripts, README, walkthrough notebook)
│   ├── 0_bigram/
│   ├── 1_gpt/
│   ├── 2_tokenizer/
│   ├── 3_modern/
│   └── 4_pretrain/     # control.py: start / pause / resume / status   (5_sft … 7_dpo to come)
├── apps/
│   ├── api/            # Flask: model serving with streaming
│   └── web/            # React + Vite: ChatGPT-style interface
├── tests/              # tests for leon/ and apps/api (pytest)
├── docs/
├── data/               # datasets and tokenizer (not versioned, rebuilt by scripts)
└── checkpoints/        # trained models, one folder per phase (not versioned)
```

**Criterio**: las fases son el *curso* y `leon/` es el *producto*.

- Las fases 0 y 1 son prototipos autocontenidos que nadie más usa.
- A partir de la fase 2, cuando un componente es definitivo y lo van a usar fases posteriores o las apps, se **gradúa** a `leon/`. La fase conserva sus scripts, su README y su notebook, y lo importa con `from leon.model import Leon`.
- Así hay una única versión del modelo, del tokenizer y del chat template, y ningún script toca `sys.path`. Es la misma organización que usa [nanochat](https://github.com/karpathy/nanochat): un paquete con el modelo más scripts por etapa más la web.

## Hoja de ruta

| Fase | Carpeta | Contenido | Hardware |
|---|---|---|---|
| 0 | `phases/0_bigram/` | Bigrama a nivel de carácter sobre *Don Quijote* | CPU |
| 1 | `phases/1_gpt/` | GPT mínimo: atención causal, multi-head, MLP, residuales, LayerNorm | RTX 5060 |
| 2 | `phases/2_tokenizer/` | Tokenizer BPE propio entrenado con Wikipedia en español | CPU |
| 3 | `phases/3_modern/` | RoPE, RMSNorm, SwiGLU, GQA, KV cache, bf16 | RTX 5060 |
| 4 | `phases/4_pretrain/` | Preentrenamiento de Leon 0.1 (~30–60M) en corpus español filtrado | RTX 5060, varias noches |
| 5 | `phases/5_sft/` | SFT conversacional (chat template, loss masking) | RTX 5060 |
| 6 | `phases/6_agent/` | SFT de tool calling + bucle agéntico | RTX 5060 |
| 7 | `phases/7_dpo/` | Alineamiento por preferencias (opcional) | RTX 5060 |

**Chat local** ([`apps/`](apps/README.md)): interfaz web al estilo de ChatGPT, con React y una API en Flask, para probar cualquier checkpoint de Leon desde la fase 3. `python apps/api/app.py` y abrir http://127.0.0.1:8000.

## Métricas comunes

Para comparar entre fases se registra siempre:

- **Loss de validación** (nats) y **bits por carácter**. A partir de la fase 2 se usará *bits por byte*, que es comparable entre tokenizers distintos.
- **Throughput** (tokens/s).
- **Muestra generada** con el prompt fijo `En un lugar de la Mancha`.

## Entorno (Windows)

Requiere Python 3.11 o posterior (probado con Python 3.14 y PyTorch 2.11.0+cu128).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
# RTX 5060 (Blackwell, sm_120): usar wheels de PyTorch con CUDA 12.8+
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[all]"      # the leon package + data, compile, api, notebooks and dev extras
```

`-e` (editable) instala `leon` apuntando a esta carpeta: los cambios en `leon/` se ven al instante, sin reinstalar. Los grupos de dependencias están en `pyproject.toml`. Por ejemplo, `pip install -e ".[api]"` instala solo lo necesario para el chat.

Para la web (`apps/web`) hace falta Node.js 20 o posterior.

Verificación:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest                        # 33 tests: tokenizer, model, data, training, run control, chat template, API, architecture
```

Calidad del código (antes de cada commit):

```powershell
ruff check .                  # lint: errores, imports sin usar, orden de imports
ruff format .                 # formato uniforme (configurado en pyproject.toml)
pytest
```

## Restricciones de hardware

Todo el proyecto se ejecuta en local: RTX 5060 (8 GB VRAM), 16 GB de RAM y sin GPUs alquiladas. Consecuencias de diseño:

- **Tamaño de Leon 0.1** (decidido en la fase 3, a partir del throughput medido): **56M de parámetros** (d = 640, 10 capas, GQA 10/5, contexto 1024) con **~2B tokens**: ~13–14 h de GPU, dos noches. Ver `phases/3_modern/README.md`.
- **Entrenamientos largos repartidos en sesiones**: todos los scripts guardan `last.pt` y aceptan `--resume`, para poder entrenar por la noche y continuar al día siguiente.
- **Datos por streaming**: los datasets se procesan por partes y se guardan ya tokenizados en disco (`np.memmap`, uint16). Nunca se cargan enteros en RAM. 2B tokens ocupan ~4 GB en disco.
- **Memoria**: bf16, gradient accumulation y, si hace falta, gradient checkpointing, para simular batches grandes en 8 GB.
- **Térmica**: en sesiones largas conviene vigilar la temperatura de la GPU (`nvidia-smi -l 5`).
