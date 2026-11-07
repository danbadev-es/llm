# Changelog

Hitos del proyecto, de la última fase a la primera. Las cifras son las medidas en una RTX 5060 Laptop (8 GB) y están detalladas en el README de cada fase.

## Sin publicar: fases 4–8 (2026-11-07)

El proyecto termina aquí por decisión del usuario: es suficiente para el objetivo educativo. El GRPO y `evaluate.py` de la fase 8 quedan como ejercicio.

### Resultados
- **Fase 8: razonamiento paso a paso** en aritmética de 5 niveles (`checkpoints/8_reasoning/`):
  - primer intento, 4.000 problemas por nivel y 2 épocas: aprendió el formato de los pasos pero no las tablas (4,5 % de aciertos), y el GRPO no tenía ninguna muestra correcta que reforzar;
  - segundo intento, 12.000 por nivel y 4 épocas, 11,7 min, val loss 0,0248 (`leon_0.1_steps`): resuelve bien paso a paso las preguntas fijas (`(25 + 38) × 4`, problemas con palabras);
  - el control sin pasos (`leon_0.1_direct`) responde `47 × 36` → 420;
  - el modelo se especializó: saluda mal.
- **Fase 7: DPO**: pares respuesta humana (elegida) frente a una muestra del propio Leon (rechazada); 1.200 preguntas × 8 muestras, 4 conservadas, en ~35 min; 4.504 pares de train y 248 de val, 281 pasos en 3,5 min. Pérdida en val 0,693 → 0,449, precisión de la recompensa 0,93. Respuestas en bucle 27 % → 16 %, terminan solas 79 % → 89 %, 88 → 62 palabras; sigue inventando datos. `checkpoints/7_dpo/leon_0.1_dpo/best.pt` es el mejor modelo de chat (`stage` `agent`, con herramientas).
- **Fase 6: herramientas**: 25.216 diálogos sintéticos, 384 pasos en 7,9 min; val loss 2,157 → 0,874. En 450 preguntas nuevas: JSON válido 100 %, herramienta correcta 100 %, respuesta correcta 92 % (calculadora 99 %, fecha 100 %, documentos 95 %, sin herramienta 99 %; "no encontrado" solo 44 %: cita pasajes irrelevantes en vez de decir que no hay nada). Falla en problemas con palabras de estructura nueva.
- **Fase 5: SFT**: 4.863 pasos, 3 épocas de 86.088 conversaciones (oasst2, Aya, Dolly, Alpaca y WildChat) a contexto 2048, desde la extensión de la fase 4. Loss de las respuestas 2,846 → **2,070** (perplejidad 7,9) en 1 h 53 min. `checkpoints/5_sft/leon_0.1_chat/best.pt` saluda, responde preguntas sencillas ("La capital de Francia es París.") y sigue inventando datos.
- **Fase 4: preentrenamiento de Leon 0.1 y extensión a 2048**:
  - 56M de parámetros, contexto 1024, 2,0B de tokens de la Wikipedia en español (15.259 pasos);
  - **val loss 2,4218 = 0,897 bits/byte** en Wikipedia y 1,687 en el Quijote (fase 3, 32M: 1,175);
  - una sesión de 12 h 32 min a ~45.500 tok/s, con 2,27 GiB de VRAM;
  - `checkpoints/4_pretrain/leon_0.1/best.pt`. Escribe español fluido con estilo de Wikipedia, pero inventa los datos.
  - **Extensión a 2048** (`leon_0.1_ctx2048`, RoPE theta 50.000, 151M de tokens, 1 h 12 min): la loss en las posiciones 1024–2047 de artículos largos queda 0,17 nats por debajo de la de 0–1023 (−0,07 antes); Quijote 1,657 bits/byte.

### Añadido
- **Fase 8, razonamiento**: `phases/8_reasoning/prepare.py` genera aritmética de 5 niveles con soluciones paso a paso (productos de una cifra, sumas de números redondos) y la versión directa como control; `train.py` (con `leon.sft`); `grpo.py` y `leon/grpo.py` (recompensa verificable, ventaja por grupo, KL hacia la referencia), probados de principio a fin en CPU; `evaluate.py` (aciertos por nivel). `control.py` con `start` y `grpo`.
- **Fase 7, DPO**: `leon/dpo.py` (pérdida DPO, log-probabilidades de una respuesta); `phases/7_dpo/prepare.py` genera los pares con el propio Leon (respuesta humana frente a la suya, varias muestras por pregunta en un batch, reanudable; `control.py prepare`); `train.py` con referencia congelada precalculada; `evaluate.py` (repeticiones, respuestas terminadas, longitud). El panel común muestra `Metrics` (precisión de la recompensa, margen).
- **Fase 6, herramientas**:
  - template con llamadas (`<|tool_call|>…<|/tool_call|><|tool_result|>…<|/tool_result|>`, `leon/chat.py`), tres herramientas (`leon/tools.py`: calculadora sin `eval`, fecha en español, búsqueda BM25 en `data/docs/`) y el bucle del agente (`leon/agent.py`);
  - `phases/6_agent/`: `prepare.py` (diálogos sintéticos con resultados de las herramientas reales, 300 documentos de ejemplo en `data/docs/ejemplo/`), `train.py` + `control.py` (sobre `leon.sft`), `evaluate.py` (JSON válido, herramienta y respuesta correctas);
  - API: `tools`/`tool_use`/`tool_result` como en Claude, y las herramientas de Leon en el servidor con `"leon": {"tools": true}`; el historial guarda las llamadas (columna `steps`, migración automática);
  - web: interruptor **Herramientas** para los modelos con `stage` `agent` y una tarjeta plegable por llamada, en su sitio dentro de la respuesta.
- **`leon/sft.py`**: el bucle de ajuste de las fases 5, 6 y 8; `leon.run.Control`, la línea de comandos común de los `control.py` de las fases 5–8.
- **Fase 5, SFT conversacional**:
  - `phases/5_sft/prepare.py`: conversaciones en español de oasst2, Aya, Dolly, Alpaca y WildChat (`allenai/WildChat-1M`, en español y sin toxicidad, multiturno reales);
  - `phases/5_sft/train.py`: SFT con loss masking y *packing*, a 2048 tokens: las conversaciones largas se recortan a sus primeros intercambios completos, las de 2 o más intercambios se repiten ×3 por época (`--multi-turn`) y el micro-batch se ajusta solo al contexto del modelo base;
  - `leon.chat.encode_chat_for_training`, `leon.data.ConversationFile` y checkpoints con `stage`;
  - `phases/5_sft/control.py` con `start`, `pause`, `resume`, `status` y `watch`; `train.py` escribe `status.json` (progreso, loss de train y val, tok/s, hora de fin, respuestas a las preguntas fijas) y se pausa limpiamente.
- **Un solo panel para todas las fases** (`leon.run.dashboard`): muestra los campos que tenga el `status.json` de cada una. Un SFT lanzado antes de este cambio se ve igual: el panel lee el progreso de las evaluaciones del `train.log`.
- **Modo chat para los modelos de conversación**: la primera visita abre un modelo de conversación en modo Chat, la fase más reciente encabeza el selector, un modelo de chat en Completar sugiere Chat y las instrucciones del sistema por defecto son las del SFT (`leon.chat.DEFAULT_SYSTEM`).
- **Indicador de contexto en la web**: anillo junto al botón de enviar, con el desglose al pasar el ratón (sistema, conversación, mensaje nuevo, reservado para la respuesta, libre) y aviso de lo que quedará fuera. Endpoint `POST /api/context`.
- **Extensión de contexto a 2048** (`phases/4_pretrain/control.py extend`):
  - `train.py --init-from --rope-theta --long-fraction`;
  - ventanas dentro de un solo artículo largo (`TokenFile.long_windows`);
  - loss por mitades del contexto en cada evaluación (`leon.training.loss_by_position`);
  - `pause`, `resume`, `status` y `watch` siguen a la última etapa iniciada.
- **Exportación a GGUF** (`leon/export.py`, `python -m leon.export`) para llama.cpp, LM Studio y Ollama, en F32, F16 o Q8_0.
  - `--verify` lo compara con los binarios oficiales de llama.cpp.
  - Con un checkpoint de la fase 4: tokenización idéntica en 210/210 textos, chat template idéntico, perplejidad a un 0,001 % (F32) y un 0,066 % (Q8_0), y generación *greedy* idéntica.
  - `tests/test_export.py`.
- **API compatible con Anthropic** (`apps/api/anthropic_api.py`):
  - `/v1/messages` (con streaming, prefill y `stop_sequences`), `/v1/messages/count_tokens`, `/v1/complete` (Text Completions) y `/v1/models`.
  - El SDK oficial `anthropic` habla con Leon cambiando solo `base_url` (`tests/test_anthropic_api.py`).
  - `top_p` (nucleus) en `Leon.stream`.
- **Web rediseñada**:
  - fluidez (pintado por frame, auto-scroll inteligente, "Ir al final", esqueletos de carga, transiciones);
  - acciones Copiar, Regenerar y Editar, avisos y reintento de errores;
  - atajos de teclado, accesibilidad completa y temas claro y oscuro;
  - usa la API `/v1`.
- `DELETE /api/conversations/<id>/messages?from=N`: regenerar o editar reemplaza el último intercambio también en el historial guardado.
- **Chat con historial**:
  - conversaciones guardadas en SQLite (`apps/api/store.py`, `data/chat.db`), con API REST `/api/conversations`;
  - el lateral muestra el historial agrupado por fecha, con renombrar y eliminar;
  - los ajustes pasan a un diálogo, y el modelo y el modo a la cabecera;
  - panel lateral en móvil;
  - la web recuerda los ajustes y el último modelo.
- La API recarga un checkpoint si el archivo cambia, así que se puede seguir en el chat un modelo que aún está entrenando.
- **Fase 4, control del preentrenamiento**: `phases/4_pretrain/control.py` con `prepare`, `start`, `pause`, `resume`, `status` y `watch` (panel en vivo con progreso, pérdidas, velocidad, hora estimada de fin, GPU, última muestra y log; `start` y `resume` lo abren en su propia ventana).
  - El entrenamiento corre como proceso independiente, con pausa limpia y `status.json`.
  - Una ejecución pausada y reanudada da **pesos idénticos** a una sin interrupciones.
- `leon/run.py`: ficheros de control de entrenamientos en segundo plano. `estimate_loss` pasa a `leon/training.py`.
- `CLAUDE.md`: qué comando corresponde a "arranca", "pausa", "continúa" y "¿cómo va?".
- `ARCHITECTURE.md`, `CONTRIBUTING.md`, este `CHANGELOG.md` y `.editorconfig`.
- `tests/test_architecture.py`: comprueba las reglas de dependencias entre `leon/`, `phases/` y `apps/`.

### Cambiado
- **Temperatura por defecto 0,5** en la web (antes 0,8): con 56M de parámetros, a 0,8 se desvía a frases sin sentido y por debajo de ~0,4 se repite. Los ajustes guardados con el valor por defecto antiguo se actualizan solos.
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

### Eliminado
- La API antigua `/api/models` y `/api/generate`, sustituida por `/v1`.

### Corregido
- Un lector de `status.json` (el panel, `status`) podía tumbar el entrenamiento en Windows, donde el renombrado atómico falla mientras otro proceso tiene el archivo abierto. Ahora se reintenta y, si sigue ocupado, se salta esa actualización.
- El entrenamiento lanzado por `control.py` abría una ventana de consola vacía, la del intérprete real que lanza el `python.exe` del venv. Cerrarla mataba el entrenamiento. Ahora corre sin ventana (`CREATE_NO_WINDOW`).

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
