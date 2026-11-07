# llm: Leon 0.1

Proyecto educativo, terminado: **Leon 0.1**, un modelo de lenguaje conversacional en español construido desde cero, con herramientas (tool calling), alineamiento por preferencias (DPO) y un experimento de razonamiento paso a paso.

Todo está implementado a mano: tokenizer, transformer, entrenamiento y chat, sin Hugging Face Transformers. Se entrena en local en una RTX 5060 Laptop (8 GB).

Cada fase vive en su propia carpeta, produce algo ejecutable y medible, y documenta en su `README.md` qué se aprendió y qué métricas obtuvo. Las fases 0–3 incluyen además un `walkthrough.ipynb` que reconstruye sus piezas paso a paso con ejemplos diminutos.

## Estado

| Fase | Estado | Resultado clave |
|---|---|---|
| 0: bigrama | ✅ | 3,327 bits/char |
| 1: GPT mínimo | ✅ | 1,627 bits/char · SDPA ×1,76 |
| 2: tokenizer BPE | ✅ | 16.384 tokens · 4,19 bytes/token en Wikipedia |
| 3: arquitectura moderna | ✅ | 1,175 bits/byte (32M, 31 min) · tamaño de Leon 0.1 fijado |
| 4: preentrenamiento | ✅ | 56M, 2,0B tokens: **0,897 bits/byte** en Wikipedia, 1,687 en el Quijote, 12,5 h · extensión a contexto 2048: 151M tokens, 1 h 12 min, la segunda mitad del contexto 0,17 nats por debajo de la primera |
| 5: SFT conversacional | ✅ | 86.088 conversaciones, contexto 2048, 3 épocas: loss de las respuestas 2,846 → **2,070** en 1 h 53 min |
| 6: herramientas y agente | ✅ | calculadora, fecha y búsqueda en documentos, 8 min: **JSON válido 100 %, herramienta correcta 100 %, respuesta correcta 92 %** (450 preguntas nuevas) |
| 7: DPO | ✅ | 4.504 pares humana / Leon: respuestas en bucle 27 % → **16 %**, terminan solas 79 % → **89 %**; sigue inventando datos · el mejor modelo de chat |
| 8: razonamiento | ✅ con pasos · GRPO sin ejecutar | aritmética en 5 niveles: con 12.000 problemas por nivel resuelve paso a paso `(25 + 38) × 4` y problemas con palabras; sin pasos, `47 × 36` → 420 · GRPO y `evaluate.py` quedan como ejercicio |

El proyecto se dio por terminado en la fase 8: es suficiente para el objetivo educativo. No queda ningún entrenamiento pendiente; el GRPO y `evaluate.py` de la fase 8 están implementados y probados de principio a fin en CPU, pero no se ejecutaron con el modelo final: quedan como ejercicio.

## Inicio rápido

Con el entorno instalado (ver [Entorno](#entorno-windows)) y los datos y checkpoints generados por las fases:

```powershell
pytest                                            # ¿todo en orden?
python phases/3_modern/sample.py --prompt "La ciudad de Madrid"
python apps/api/app.py                            # chat en http://127.0.0.1:8000
```

En el chat, el mejor modelo es **`7_dpo/leon_0.1_dpo`** (SFT + herramientas + DPO): modo Chat y, si se quiere, el interruptor **Herramientas** (calculadora, fecha y búsqueda en `data/docs/`). Elígelo en la cabecera: la primera visita abre el de la fase 5. `8_reasoning/leon_0.1_steps` resuelve aritmética paso a paso, pero se especializó y conversa peor.

## Documentación

| Documento | Para qué |
|---|---|
| [`docs/fundamentos.md`](docs/fundamentos.md) | Conceptos desde cero, objetivo y criterios de éxito de Leon 0.1 |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Componentes, flujo de datos, formatos compartidos, decisiones de diseño y reglas de dependencias |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Entorno, flujo de trabajo, convenciones y cómo añadir una fase |
| [`CHANGELOG.md`](CHANGELOG.md) | Hitos y resultados, fase a fase |
| [`CLAUDE.md`](CLAUDE.md) | Instrucciones para sesiones de Claude Code: qué comando corresponde a "arranca", "pausa", "continúa"... |
| `phases/N_*/README.md` | Cada fase: conceptos, ejecución, ejercicios y resultados medidos: [0_bigram](phases/0_bigram/README.md), [1_gpt](phases/1_gpt/README.md), [2_tokenizer](phases/2_tokenizer/README.md), [3_modern](phases/3_modern/README.md), [4_pretrain](phases/4_pretrain/README.md), [5_sft](phases/5_sft/README.md), [6_agent](phases/6_agent/README.md), [7_dpo](phases/7_dpo/README.md), [8_reasoning](phases/8_reasoning/README.md) |
| [`apps/README.md`](apps/README.md) | Chat local: API, web, modos y seguridad |

## Usar Leon en LM Studio y Ollama

`leon.export` convierte un checkpoint a **GGUF**, el formato de llama.cpp, que es el motor de LM Studio y Ollama:

```powershell
pip install -e ".[export]"
python -m leon.export checkpoints/7_dpo/leon_0.1_dpo/best.pt leon-0.1-chat-q8_0.gguf --dtype q8_0   # el mejor modelo de chat
# opcional: comprobarlo con los binarios de llama.cpp (https://github.com/ggml-org/llama.cpp/releases)
python -m leon.export checkpoints/7_dpo/leon_0.1_dpo/best.pt leon-0.1-chat-q8_0.gguf --dtype q8_0 --verify C:\ruta\a\llama.cpp
```

- **Qué se exporta**: la arquitectura `llama` de llama.cpp, el tokenizer v2 con su regla `llama-bpe` y, en los modelos de chat, el chat template en Jinja con `<|end|>` como fin de turno.
- **Tamaño**: ~57 MB en Q8_0 y ~107 MB en F16.
- **Verificación** (`--verify`), medida con un checkpoint de la fase 4:
  - la tokenización de llama.cpp es idéntica en 210/210 textos (Wikipedia, el Quijote y casos límite);
  - el chat template produce los mismos tokens que `leon.chat`;
  - la perplejidad difiere un 0,001 % en F32 y un 0,066 % en Q8_0;
  - la generación *greedy* es idéntica token a token.
- **LM Studio**: copia el `.gguf` a su carpeta de modelos. Lee el chat template del propio archivo.
- **Herramientas**: el bucle del agente (`leon.agent`) solo existe en `apps/api`; en LM Studio y Ollama el modelo conversa, pero nadie ejecuta sus llamadas.
- **Ollama**: necesita un `Modelfile` con la plantilla en su formato. Sin probar:

  ```
  FROM ./leon-0.1-chat-q8_0.gguf
  TEMPLATE """{{- range .Messages }}<|{{ .Role }}|>{{ .Content }}<|end|>{{ end }}<|assistant|>"""
  PARAMETER stop "<|end|>"
  PARAMETER stop "<|user|>"
  ```

  y después `ollama create leon -f Modelfile`.

## Estructura del repositorio

```
llm/
├── pyproject.toml      # makes the `leon` package installable: pip install -e .
├── leon/               # the product: reusable, tested code shared by phases and apps
│   ├── tokenizer.py    #   byte-level BPE                         (built in phase 2)
│   ├── model.py        #   Leon: RoPE, RMSNorm, SwiGLU, GQA, KV cache (built in phase 3)
│   ├── data.py         #   token files with np.memmap             (built in phase 3)
│   ├── training.py     #   LR schedule, eval, bits/byte, atomic checkpoints with RNG state (phase 3)
│   ├── run.py          #   background runs: pause, status.json, pid, dashboard, Control CLI (phase 4)
│   ├── chat.py         #   chat template, with tool calls          (phases 5–6 and apps/api)
│   ├── sft.py          #   the supervised fine-tuning loop          (phases 5, 6 and 8)
│   ├── tools.py        #   calculadora, fecha_hora, buscar_documentos (BM25) (phase 6)
│   ├── agent.py        #   the agent loop: generate, parse the call, run it, continue (phase 6)
│   ├── dpo.py          #   DPO loss and answer log-probabilities  (phase 7)
│   ├── grpo.py         #   verifiable reward, group advantages, GRPO loss (phase 8)
│   ├── export.py       #   GGUF for llama.cpp, LM Studio and Ollama
│   └── paths.py        #   where data/ and checkpoints/ live
├── phases/             # the course: one chapter per phase (scripts, README; notebook in phases 0–3)
│   ├── 0_bigram/
│   ├── 1_gpt/
│   ├── 2_tokenizer/
│   ├── 3_modern/
│   ├── 4_pretrain/     # control.py: start / pause / resume / status / watch / extend
│   ├── 5_sft/          # prepare.py, control.py: conversational SFT
│   ├── 6_agent/        # prepare.py, control.py, evaluate.py: tool calling
│   ├── 7_dpo/          # prepare.py (GPU), control.py, evaluate.py: preference learning
│   └── 8_reasoning/    # prepare.py, control.py (start / grpo), grpo.py, evaluate.py: step-by-step arithmetic
├── apps/
│   ├── api/            # Flask: Anthropic-compatible /v1 API, streaming, tools, history (SQLite)
│   └── web/            # React + Vite + Tailwind v4 + shadcn/ui: ChatGPT-style interface
├── tests/              # tests for leon/ and apps/api (pytest)
├── docs/               # fundamentos.md: concepts, goal and success criteria
├── data/               # datasets and tokenizer (not versioned, rebuilt by scripts)
└── checkpoints/        # trained models, one folder per phase (not versioned)
```

**Criterio**: las fases son el *curso* y `leon/` es el *producto*.

- Las fases 0 y 1 son prototipos autocontenidos que nadie más usa.
- A partir de la fase 2, cuando un componente es definitivo y lo van a usar fases posteriores o las apps, se **gradúa** a `leon/`. La fase conserva sus scripts y su README (y su notebook, si lo tiene), y lo importa con `from leon.model import Leon`.
- Así hay una única versión del modelo, del tokenizer y del chat template, y ningún script toca `sys.path`. Es la misma organización que usa [nanochat](https://github.com/karpathy/nanochat): un paquete con el modelo más scripts por etapa más la web.

## Hoja de ruta

| Fase | Carpeta | Contenido | Hardware |
|---|---|---|---|
| 0 | `phases/0_bigram/` | Bigrama a nivel de carácter sobre *Don Quijote* | CPU |
| 1 | `phases/1_gpt/` | GPT mínimo: atención causal, multi-head, MLP, residuales, LayerNorm | RTX 5060 |
| 2 | `phases/2_tokenizer/` | Tokenizer BPE propio entrenado con Wikipedia en español | CPU |
| 3 | `phases/3_modern/` | RoPE, RMSNorm, SwiGLU, GQA, KV cache, bf16 | RTX 5060 |
| 4 | `phases/4_pretrain/` | Preentrenamiento de Leon 0.1 (56M) con 2,0B tokens de la Wikipedia en español, y extensión del contexto a 2048 | RTX 5060, 12,5 h + 1 h 12 min |
| 5 | `phases/5_sft/` | SFT conversacional (chat template, loss masking, packing) | RTX 5060, 1 h 53 min |
| 6 | `phases/6_agent/` | SFT de tool calling + bucle agéntico | CPU (datos) + RTX 5060, 7,9 min |
| 7 | `phases/7_dpo/` | Alineamiento por preferencias (DPO) con pares respuesta humana / respuesta de Leon | RTX 5060, ~35 min (pares) + 3,5 min |
| 8 | `phases/8_reasoning/` | Razonamiento paso a paso en aritmética: SFT con cadenas de pasos y aprendizaje por refuerzo con recompensa verificable (GRPO). Se compara con un control sin pasos | CPU (datos) + RTX 5060, 11,7 min (GRPO sin ejecutar) |

**Chat local** ([`apps/`](apps/README.md)): interfaz web al estilo de ChatGPT, con React y una API en Flask, para probar cualquier checkpoint de Leon desde la fase 3, con herramientas para los modelos de las fases 6–8. `python apps/api/app.py` y abrir http://127.0.0.1:8000.

## Métricas comunes

Para comparar entre fases se registra siempre:

- **Loss de validación** (nats) y **bits por carácter** en las fases 0 y 1; desde la fase 3, *bits por byte*, que es comparable entre tokenizers distintos.
- Desde la fase 5, **loss de las respuestas** (solo los tokens del asistente) y respuestas a unas **preguntas fijas** en cada evaluación; las fases 6–8 añaden su `evaluate.py`.
- **Throughput** (tokens/s).
- **Muestra generada** con el prompt fijo `En un lugar de la Mancha` (modelos base).

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
pytest                        # 78 tests, CPU: leon/ (tokenizer, model, data, training, run, chat, sft, tools, agent, DPO, GRPO, export), API, architecture
```

Calidad del código (antes de cada commit):

```powershell
ruff check .                  # lint: errores, imports sin usar, orden de imports
ruff format .                 # formato uniforme (configurado en pyproject.toml)
pytest
python phases/1_gpt/test_model.py
cd apps/web; npm run build; cd ../..   # si cambió la web
```

## Restricciones de hardware

Todo el proyecto se ejecuta en local: RTX 5060 (8 GB VRAM), 16 GB de RAM y sin GPUs alquiladas. Consecuencias de diseño:

- **Tamaño de Leon 0.1** (decidido en la fase 3, a partir del throughput medido): **56M de parámetros** (d = 640, 10 capas, GQA 10/5, contexto 1024) con **~2B tokens**: ~13–14 h de GPU estimadas en la fase 3; el preentrenamiento real tardó 12,5 h. Ver `phases/3_modern/README.md`.
- **Entrenamientos largos repartidos en sesiones**: todos los scripts guardan `last.pt` y aceptan `--resume`, y cada entrenamiento largo (fases 4–8) tiene un `control.py` con `start / pause / resume / status / watch` que lo lanza en segundo plano, para poder entrenar por la noche y continuar al día siguiente.
- **Datos por streaming**: los datasets se procesan por partes y se guardan ya tokenizados en disco (`np.memmap`, uint16). Nunca se cargan enteros en RAM. 2B tokens ocupan ~4 GB en disco.
- **Memoria**: bf16, gradient accumulation y, si hace falta, gradient checkpointing, para simular batches grandes en 8 GB.
- **Térmica**: en sesiones largas conviene vigilar la temperatura de la GPU (`nvidia-smi -l 5`).
