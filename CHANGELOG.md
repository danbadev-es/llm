# Changelog

Hitos del proyecto, de la última fase a la primera. Las cifras son las medidas en una RTX 5060 Laptop (8 GB) y están detalladas en el README de cada fase.

## Sin publicar

### Resultados
- **Fase 4: preentrenamiento de Leon 0.1 y extensión a 2048 terminados** (2026-11-07):
  - 56M de parámetros, contexto 1024, 2,0B de tokens de la Wikipedia en español (15.259 pasos);
  - **val loss 2,4218 = 0,897 bits/byte** en Wikipedia y 1,687 en el Quijote (fase 3, 32M: 1,175);
  - una sesión de 12 h 32 min a ~45.500 tok/s, con 2,27 GiB de VRAM;
  - `checkpoints/4_pretrain/leon_0.1/best.pt`. Escribe español fluido con estilo de Wikipedia, pero inventa los datos.
  - **Extensión a 2048** (`leon_0.1_ctx2048`, 151M de tokens, 1 h 12 min): la loss en las posiciones 1024–2047 de artículos largos queda 0,17 por debajo de la de 0–1023 (−0,07 antes); Quijote 1,657 bits/byte.

### Añadido
- **Modo chat listo para los modelos de la fase 5**: la primera visita abre el modelo de conversación más reciente en modo Chat, la fase más reciente encabeza el selector, un modelo de chat en Completar sugiere Chat y las instrucciones del sistema por defecto son las del SFT (`leon.chat.DEFAULT_SYSTEM`).
- **Fase 5 con control y panel**: `phases/5_sft/control.py` con `start`, `pause`, `resume`, `status` y `watch`. `train.py` escribe `status.json` (progreso, loss de train y val, tok/s, hora de fin, respuestas a las preguntas fijas) y se pausa limpiamente.
  - El panel es uno solo para todas las fases (`leon.run.dashboard`): muestra los campos que tenga el `status.json` de cada una. La fase 4 lo usa también.
  - Un SFT lanzado antes de este cambio se ve igual: el panel lee el progreso de las evaluaciones del `train.log`.
- **Fase 8 en la hoja de ruta: razonamiento** paso a paso en aritmética, con SFT de cadenas de pasos y refuerzo con recompensa verificable (GRPO). Explicado en `docs/fundamentos.md`.
- **Indicador de contexto en la web**: anillo junto al botón de enviar, con el desglose al pasar el ratón (sistema, conversación, mensaje nuevo, reservado para la respuesta, libre) y aviso de lo que quedará fuera. Endpoint `POST /api/context`.
- **Extensión de contexto a 2048** (`control.py extend`, para después del preentrenamiento):
  - `train.py --init-from --rope-theta --long-fraction`;
  - ventanas dentro de un solo artículo largo (`TokenFile.long_windows`);
  - loss por mitades del contexto en cada evaluación (`leon.training.loss_by_position`);
  - `pause`, `resume`, `status` y `watch` siguen a la última etapa iniciada.
- **SFT a 2048 tokens**:
  - las conversaciones largas se recortan a sus primeros intercambios completos;
  - WildChat (`allenai/WildChat-1M`, en español y sin toxicidad) aporta conversaciones multiturno reales;
  - las conversaciones de 2 o más intercambios se repiten ×3 por época (`--multi-turn`);
  - el micro-batch se ajusta solo al contexto del modelo base.
- **Exportación a GGUF** (`leon/export.py`, `python -m leon.export`) para llama.cpp, LM Studio y Ollama, en F32, F16 o Q8_0.
  - `--verify` lo compara con los binarios oficiales de llama.cpp.
  - Con un checkpoint de la fase 4: tokenización idéntica en 210/210 textos, chat template idéntico, perplejidad a un 0,001 % (F32) y un 0,066 % (Q8_0), y generación *greedy* idéntica.
  - `tests/test_export.py`.

### Cambiado
- **La web usa Tailwind CSS v4 y shadcn/ui**:
  - el tema son las variables de shadcn con la paleta de Leon, en claro y oscuro;
  - los menús, diálogos, selector de modelo, panel lateral del móvil, tooltips y avisos (Sonner) son componentes de shadcn sobre Radix, en lugar de los hechos a mano;
  - los componentes los genera el CLI oficial (`npx shadcn@latest add`) y no se editan;
  - iconos de `lucide-react`;
  - se eliminan `styles.css` (1.500 líneas) y los componentes propios `Menu`, `Modal`, `Toasts` e `Icon`.
- **Tokenizer v2**: mismo vocabulario de 16k, regla de pre-tokenización de Llama 3, idéntica a la de llama.cpp, para poder exportar Leon a GGUF (LM Studio y Ollama).
  - Cuesta un 1,8 % de compresión en Wikipedia.
  - Cada checkpoint guarda su versión de tokenizer.
  - Se re-tokenizaron Wikipedia, el Quijote y los datos de SFT, y **la fase 4 se reinició desde cero**: llevaba un 10 %.

### Añadido
- **API compatible con Anthropic** (`apps/api/anthropic_api.py`):
  - `/v1/messages` (con streaming, prefill y `stop_sequences`), `/v1/messages/count_tokens`, `/v1/complete` (Text Completions) y `/v1/models`.
  - El SDK oficial `anthropic` habla con Leon cambiando solo `base_url` (`tests/test_anthropic_api.py`).
  - `top_p` (nucleus) en `Leon.stream`.
- **Web rediseñada**:
  - fluidez (pintado por frame, auto-scroll inteligente, "Ir al final", esqueletos de carga, transiciones);
  - acciones Copiar, Regenerar y Editar, avisos y reintento de errores;
  - atajos de teclado, accesibilidad completa y temas claro y oscuro.
  - Usa la API `/v1`.
- `DELETE /api/conversations/<id>/messages?from=N`: regenerar o editar reemplaza el último intercambio también en el historial guardado.
- **Fase 5 (en preparación)**:
  - `phases/5_sft/prepare.py`: 74k conversaciones en español de oasst2, Aya, Dolly y Alpaca.
  - `phases/5_sft/train.py`: SFT con loss masking y *packing*.
  - `leon.chat.encode_chat_for_training` y `leon.data.ConversationFile`.
  - Checkpoints con `stage`.

### Eliminado
- La API antigua `/api/models` y `/api/generate`, sustituida por `/v1`.

- `control.py watch`: panel en vivo del preentrenamiento, con progreso, pérdidas, velocidad, hora estimada de fin, GPU, última muestra y log. `start` y `resume` lo abren en su propia ventana.

### Corregido
- El entrenamiento lanzado por `control.py` abría una ventana de consola vacía, la del intérprete real que lanza el `python.exe` del venv. Cerrarla mataba el entrenamiento. Ahora corre sin ventana (`CREATE_NO_WINDOW`).

### Añadido (antes)
- **Chat con historial**:
  - Conversaciones guardadas en SQLite (`apps/api/store.py`, `data/chat.db`), con API REST `/api/conversations`.
  - El lateral muestra el historial agrupado por fecha, con renombrar y eliminar.
  - Los ajustes pasan a un diálogo, y el modelo y el modo a la cabecera.
  - Panel lateral en móvil.
  - La web recuerda los ajustes y el último modelo.
- La API recarga un checkpoint si el archivo cambia, así que se puede seguir en el chat un modelo que aún está entrenando.
- **Fase 4, lista para arrancar**: `phases/4_pretrain/control.py` con `prepare`, `start`, `pause`, `resume` y `status`.
  - El entrenamiento corre como proceso independiente, con pausa limpia y `status.json`.
  - Una ejecución pausada y reanudada da **pesos idénticos** a una sin interrupciones.
- `leon/run.py`: ficheros de control de entrenamientos en segundo plano. `estimate_loss` pasa a `leon/training.py`.
- `CLAUDE.md`: qué comando corresponde a "arranca", "pausa", "continúa" y "¿cómo va?".
- `ARCHITECTURE.md`, `CONTRIBUTING.md`, este `CHANGELOG.md` y `.editorconfig`.
- `tests/test_architecture.py`: comprueba las reglas de dependencias entre `leon/`, `phases/` y `apps/`.

## Fase 3: arquitectura moderna (2026-11-06)

### Añadido
- `leon/model.py`: RMSNorm, RoPE, SwiGLU, GQA, KV cache y generación en streaming.
- `leon/data.py`: lectura de los `.bin` con `np.memmap`. `leon/training.py`: checkpoints atómicos con estado del RNG.
- `phases/3_modern/benchmark.py`: throughput de los tamaños candidatos.
- `apps/api` (Flask, Server-Sent Events) y `apps/web` (React + Vite): chat local con modos *completar* y *chat*.

### Resultados
- Modelo de 32M en Wikipedia: **1,175 bits/byte** en 31 minutos, a 70.620 tok/s sostenidos.
- **Tamaño de Leon 0.1 decidido**: 56M (d = 640, 10 capas, GQA 10/5, contexto 1024), ~2B tokens. Entrena a 41.946 tok/s, ~13–14 h.
- `torch.compile` da ×1,5–1,6. Los micro-batches de más de 4.096 tokens desbordan la VRAM por los logits de un vocabulario de 16k.

### Corregido (revisión de código)
- `--resume` repetía los batches del principio y podía cambiar el contexto sin avisar.
- El último token de cada fichero nunca se usaba como objetivo en `get_batch`.
- La generación con KV cache se quedaba un token corta.
- API:
  - validación completa antes de empezar el streaming;
  - recorte de contexto por turnos completos;
  - protección contra *DNS rebinding* y *path traversal*.

## Fase 2: tokenizer BPE (2026-11-06)

### Añadido
- `leon/tokenizer.py`: BPE a nivel de byte, entrenamiento incremental y 16 tokens especiales reservados (chat y herramientas).
- Pipeline de datos: Wikipedia en español (shard 0) en JSONL, y tokenización en paralelo a `.bin` uint16.

### Resultados
- 16.384 tokens entrenados en 30 s. Comprimen **4,19 bytes/token** en Wikipedia y 3,50 en el Quijote.
- Shard 0: 282M tokens. Los 13 shards darían ~1,4B.

## Fase 1: GPT mínimo (2026-11-06)

### Resultados
- 10,78M de parámetros, **1,627 bits/char** (el bigrama daba 3,327).
- SDPA frente a atención manual: ×1,76 de velocidad (143k frente a 81k tok/s) y un 38 % menos de VRAM.

## Fase 0: bigrama

### Resultados
- Bigrama: val 2,3047 por conteo y 2,3061 con gradiente, **3,327 bits/char**. 1,39M tok/s en CPU.
