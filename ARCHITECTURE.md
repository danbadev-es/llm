# Arquitectura

Mapa técnico del repositorio: qué piezas hay, cómo fluyen los datos entre ellas, qué formatos comparten y qué reglas no deben romperse. Para la motivación y los conceptos de fondo, ver [`docs/fundamentos.md`](docs/fundamentos.md). Para el detalle y los resultados de cada etapa, ver el `README.md` de cada fase.

## Visión general

El repositorio tiene dos mitades con propósitos distintos:

- **El curso** (`phases/`): una carpeta por fase, que se lee en orden. Cada fase contiene sus scripts y un README con resultados medidos; las fases 0–3, además, un notebook que reconstruye las piezas paso a paso.
- **El producto** (`leon/` y `apps/`): el código definitivo de Leon 0.1, que se reutiliza, está testeado y es estable.

Cuando una fase produce un componente que usarán fases posteriores o las apps, ese componente se **gradúa** a `leon/`. La fase conserva los scripts que lo entrenan, lo miden y lo explican. Las fases 0 y 1 son prototipos autocontenidos: no dependen de `leon/` y nada depende de ellas.

```mermaid
flowchart LR
    subgraph course["Curso: phases/"]
        P0["0_bigram"]
        P1["1_gpt"]
        P2["2_tokenizer"]
        P3["3_modern"]
        P4["4_pretrain"]
        P5["5_sft"]
        P6["6_agent"]
        P7["7_dpo"]
        P8["8_reasoning"]
    end
    subgraph product["Producto"]
        L["leon/<br/>paquete Python"]
        API["apps/api<br/>Flask"]
        WEB["apps/web<br/>React + Vite"]
    end
    P2 -- "gradúa tokenizer" --> L
    P3 -- "gradúa modelo, datos, entrenamiento" --> L
    P4 -- "gradúa run (control en segundo plano)" --> L
    P5 -- "gradúa sft, chat para entrenar" --> L
    P6 -- "gradúa tools, agent" --> L
    P7 -- "gradúa dpo" --> L
    P8 -- "gradúa grpo" --> L
    API -- "importa" --> L
    WEB -- "HTTP /v1 y /api" --> API
```

## Flujo de datos

```mermaid
flowchart TD
    W["Wikipedia ES<br/>parquet, 13 shards"] -->|"phases/2_tokenizer/prepare.py"| J["data/wiki_es/{train,val}.jsonl<br/>NFC, split 90/10 por artículo"]
    Q["Don Quijote<br/>Project Gutenberg"] -->|"phases/0_bigram/prepare.py"| QT["data/quijote/input.txt"]
    J -->|"train_tokenizer.py<br/>100M caracteres"| TK["data/tokenizer/leon_bpe.json<br/>16.384 tokens"]
    J & QT -->|"encode.py<br/>pool de procesos"| B["*.bin uint16 + *.json<br/>(tokens, bytes)"]
    TK --> B
    B -->|"np.memmap + get_batch"| TR["phases/N/train.py"]
    TR -->|"atómico, con RNG"| C["checkpoints/N_fase/run/<br/>last.pt · best.pt"]
    C -->|"--base / --init-from"| TR
    C -->|"engine.py"| API2["apps/api"]
    TK --> API2
    API2 -->|"Server-Sent Events"| UI["apps/web"]
```

Desde la fase 5, cada fase parte del mejor modelo de la anterior y tiene sus propios datos, que prepara su `prepare.py`:

| Fase | Parte de | Datos | Checkpoint (`stage`) |
|---|---|---|---|
| 4 | nada (desde cero); la extensión, de `4_pretrain/leon_0.1` | `data/wiki_es/*.bin` | `4_pretrain/leon_0.1`, `leon_0.1_ctx2048` (`base`) |
| 5 | `4_pretrain/leon_0.1_ctx2048` | `data/sft/{train,val}.npz` (conversaciones empaquetadas) | `5_sft/leon_0.1_chat` (`chat`) |
| 6 | `5_sft/leon_0.1_chat` | `data/agent/*.npz` (diálogos sintéticos), `data/docs/` | `6_agent/leon_0.1_agent` (`agent`) |
| 7 | `6_agent/leon_0.1_agent` | `data/dpo/{train,val}.jsonl` (pares que genera el propio Leon en la GPU) | `7_dpo/leon_0.1_dpo` (`agent`) |
| 8 | `7_dpo/leon_0.1_dpo` | `data/reasoning/{steps,direct}/*.npz` | `8_reasoning/leon_0.1_steps`, `leon_0.1_direct` (`agent`) |

`data/` y `checkpoints/` no se versionan: todo se reconstruye con los scripts de las fases.

## El paquete `leon/`

| Módulo | Responsabilidad | Depende de |
|---|---|---|
| `paths.py` | Rutas de `data/`, `checkpoints/` y de las versiones del tokenizer | — |
| `tokenizer.py` | `BPETokenizer`: pre-tokenización por regex, entrenamiento BPE incremental, `encode`/`decode`, tokens especiales, `save`/`load` | — |
| `model.py` | `Leon` y `LeonConfig`: RMSNorm, RoPE, atención con GQA, SwiGLU, `KVCache`, `stream`/`generate` | torch |
| `data.py` | `TokenFile` (`.bin` con `np.memmap`, `get_batch`), `load_splits`, `load_tokenizer` | `paths`, `tokenizer` |
| `training.py` | `lr_at`, `estimate_loss`, `loss_by_position` (loss por mitades del contexto), `bits_per_byte`, `save_checkpoint`/`load_checkpoint` | `model`, `data`, `paths` |
| `run.py` | `RunFiles`: ficheros de control de un entrenamiento en segundo plano (`PAUSE`, `status.json`, `train.pid`, `train.log`), `is_process_alive`, `keep_system_awake`. Lo común a los `control.py` de todas las fases: `launch` (proceso sin ventana), `wait_for_pause`, el panel en vivo (`dashboard`, `watch`), que muestra los campos que tenga el `status.json` de cada fase, y `Control`, la línea de comandos `start / pause / resume / status / watch`. `status.json` se escribe de forma atómica y, si en Windows un lector lo tiene abierto, se reintenta y en último caso se salta esa actualización | `paths` |
| `chat.py` | `encode_chat` y `encode_chat_for_training`: el chat template, con llamadas a herramientas desde la fase 6; `tool_call_json`, `DEFAULT_SYSTEM` | `tokenizer` |
| `sft.py` | El bucle de ajuste supervisado de las fases 5, 6 y 8: `add_sft_args`, `micro_batch_plan` (conversaciones empaquetadas), `validation_loss`, `answer` (preguntas fijas) y `train_sft`, con estado, pausa y checkpoints | `chat`, `data`, `model`, `run`, `training` |
| `tools.py` | Las herramientas de Leon (fase 6): `calculadora` (evaluador aritmético sobre el AST, sin `eval`), `fecha_hora`, `buscar_documentos` (BM25 sobre `data/docs/`), y `system_prompt`, que las anuncia al modelo | `chat`, `paths` |
| `dpo.py` | DPO (fase 7): `encode_answer`, `batch` (secuencias con máscara de respuesta), `sequence_logprobs` y `dpo_loss` con sus métricas | `chat` |
| `grpo.py` | GRPO (fase 8): `reward` verificable, `group_advantages`, `token_logprobs` y `grpo_loss` con KL | `dpo` |
| `agent.py` | `run_agent`: el bucle del agente (generar, parsear la llamada, ejecutar, añadir el resultado, continuar); `parse_call` | `tools` |
| `export.py` | GGUF para llama.cpp, LM Studio y Ollama (`python -m leon.export`); `--verify` lo compara con los binarios de llama.cpp | `model`, `data`, `chat`, `tokenizer`; extra opcional `gguf`, `jinja2` |

### El modelo

```mermaid
flowchart TD
    I["ids (batch, T)"] --> E["token_embedding<br/>(sin embedding de posición)"]
    E --> B1
    subgraph B1["Block × n_layer"]
        direction TB
        N1["RMSNorm"] --> A["Attention<br/>q, k con RoPE · GQA · SDPA causal<br/>+ KVCache al generar"]
        A --> R1(("+"))
        R1 --> N2["RMSNorm"] --> M["SwiGLU<br/>down(silu(gate) · up)"]
        M --> R2(("+"))
    end
    B1 --> NF["RMSNorm final"] --> H["lm_head<br/>(pesos compartidos con el embedding)"] --> O["logits (batch, T, vocab)"]
```

Configuración de Leon 0.1 (fijada en la fase 3 midiendo el throughput): `n_embd=640`, `n_layer=10`, `n_head=10`, `n_kv_head=5`, `block_size=1024`, `vocab_size=16384`. Son 56M de parámetros, 45,5M sin el embedding.

## Formatos compartidos (contratos)

Estos formatos los producen unas piezas y los consumen otras. Cambiarlos rompe datos o checkpoints existentes.

### Vocabulario

| ids | Contenido |
|---|---|
| 0–255 | Los 256 bytes. Cualquier texto es codificable y no existe el token desconocido |
| 256–16.367 | Los 16.112 merges, en el orden en que se aprendieron (el id es también la prioridad al codificar) |
| 16.368–16.383 | 16 tokens especiales: `<\|endoftext\|>` 16368, `<\|system\|>` 16369, `<\|user\|>` 16370, `<\|assistant\|>` 16371, `<\|end\|>` 16372, `<\|tool_call\|>` 16373, `<\|/tool_call\|>` 16374, `<\|tool_result\|>` 16375, `<\|/tool_result\|>` 16376, `<\|reserved_0..6\|>` |

Los especiales se reservaron en la fase 2 para que el vocabulario no cambie entre el preentrenamiento (fase 4), el SFT (fase 5) y el tool calling (fase 6).

**Versiones del tokenizer** (`leon/paths.py`):

- **v1**: el primero, con regla de pre-tokenización propia; lo usan los modelos de la fase 3.
- **v2**: el de Leon 0.1 desde la fase 4. Mismo vocabulario, regla de pre-tokenización de Llama 3, idéntica a la de llama.cpp (`llama-bpe`), así que un export a GGUF tokeniza igual en LM Studio y Ollama.

Cada checkpoint guarda su versión, y los datos de SFT también, de modo que el SFT se niega a mezclar versiones.

### Ficheros de tokens (`*.bin` + `*.json`)

- **`.bin`**: ids en uint16 contiguos, sin cabecera. Por eso el vocabulario no puede pasar de 65.536 tokens.
- **`.json`**: `{"tokens", "bytes", "documents", "bytes_per_token", "tokenizer"}`. `bytes` cuenta solo el texto original; `<|endoftext|>` cuenta como 0 bytes.
- **En Wikipedia**, cada artículo va seguido de `<|endoftext|>`.
- **Split**: 1 de cada 10 artículos va a val. En el Quijote es el mismo corte 90/10 por caracteres que en las fases 0 y 1.

### Checkpoints (`checkpoints/<N_fase>/<run>/{last,best}.pt`)

| Clave | Contenido |
|---|---|
| `model` | `state_dict` |
| `config` | `asdict(LeonConfig)`. La arquitectura sale siempre de aquí, nunca de la línea de comandos |
| `tokenizer` | Versión del tokenizer (`v1`, `v2`). Si falta, el checkpoint es anterior a esta clave y usa `v1` |
| `step`, `best_val` | Progreso |
| `optimizer` | Solo en `last.pt`. `best.pt` no lo necesita y ocupa un tercio |
| `rng_cpu`, `rng_cuda` | Estado de los generadores aleatorios, para que `--resume` continúe con los mismos batches |
| `settings` | Los argumentos de la ejecución (lr, dataset, pasos...) |
| `stage` | `"base"` (fase 4), `"chat"` (fase 5) o `"agent"` (fase 6, sabe llamar a herramientas). Las fases 7 y 8 heredan el de su modelo de partida, así que sus modelos son `"agent"`. Si falta (fases 1–3), la API lo trata como `"base"` |

Se escriben de forma atómica: primero un `.tmp` y luego `os.replace`.

### Chat template

```
<|system|>…<|end|><|user|>…<|end|><|assistant|>…<|end|><|user|>…<|end|><|assistant|>
```

Con herramientas (fase 6), la respuesta escribe la llamada en JSON y sigue en el mismo turno tras el resultado, que inserta el *runtime*:

```
<|assistant|><|tool_call|>{"name": "calculadora", "arguments": {"expresion": "17*24"}}<|/tool_call|><|tool_result|>408<|/tool_result|>Son 408.<|end|>
```

- **Como mensajes**: un mensaje `assistant` con `tool_calls: [{"name", "arguments"}]`, un `{"role": "tool", "content"}` por llamada y otro `assistant` con el resto de la respuesta.
- **Qué se aprende**: el texto de las respuestas, las llamadas (marcadores y JSON) y `<|end|>`; nunca el resultado, que escribe el *runtime*.
- **JSON**: siempre `{"name": …, "arguments": …}` en ese orden, sin escapar los acentos (`leon.chat.tool_call_json`).

Lo define `leon.chat`, y es la única implementación: las fases 5–8 lo usan para entrenar, `leon.export` para el chat template del GGUF y `apps/api` para la inferencia.

### GGUF (`leon/export.py`)

| Leon | GGUF (arquitectura `llama` de llama.cpp) |
|---|---|
| `token_embedding` | `token_embd` (también la capa de salida: no se escribe `output.weight`) |
| `blocks.i.attn_norm`, `attn.{q,k,v}_proj`, `attn.out_proj` | `blk.i.attn_norm`, `attn_{q,k,v}`, `attn_output` |
| `blocks.i.mlp_norm`, `mlp.{gate,up,down}_proj` | `blk.i.ffn_norm`, `ffn_{gate,up,down}` |
| `final_norm` | `output_norm` |

- **Sin permutar pesos**: el RoPE de Leon rota pares de canales contiguos, que es el modo "normal" de llama.cpp. Los Llama de Hugging Face rotan las dos mitades de la cabeza, y por eso su conversor permuta q y k.
- **Tokenizer**: modelo `gpt2` (BPE a nivel de byte, con el mapa de bytes de GPT-2) y regla `llama-bpe`, que es la del v2. Los tokens especiales van como tokens de control.
- **Sin token de inicio**: `add_bos_token = false`, porque Leon no usa BOS.

### API HTTP

Compatible con la API de Claude (`apps/api/anthropic_api.py`), para que el SDK oficial `anthropic` hable con Leon cambiando solo la URL base: `POST /v1/messages` (modo chat, con o sin streaming), `POST /v1/messages/count_tokens`, `POST /v1/complete` (modo completar, la antigua Text Completions) y `GET /v1/models`. Acepta herramientas como la de Claude (`tools` → bloques `tool_use`; `tool_result` en el siguiente mensaje), y las de Leon se ejecutan en el servidor con `"leon": {"tools": true}`. Lo propio de Leon va en una clave `leon` (estadísticas de generación, metadatos del checkpoint) y en la cabecera `X-Leon-Conversation` (guardar en el historial). Además, el CRUD de `/api/conversations` y `POST /api/context` (el indicador de contexto de la web). La web usa esta misma API (`apps/web/src/api.js` traduce entre ella y sus componentes). Ver [`apps/README.md`](apps/README.md).

### Historial de conversaciones

SQLite en `data/chat.db` (`apps/api/store.py`):

- **Tablas**: `conversations` (id, título, modo, modelo, fechas) y `messages` (rol, contenido, `prompt` en las respuestas del modo completar, `stats` en JSON, `error`, y `steps` en JSON con las llamadas a herramientas: `[{"id", "name", "input", "result", "at"}]`, donde `at` es cuánto texto de la respuesta iba antes).
- **Migraciones**: al abrir una base de datos anterior, `ConversationStore` añade las columnas que falten (`steps`).
- **Borrado**: en cascada; al eliminar una conversación se eliminan sus mensajes.
- **Conexiones**: una por petición, porque Flask atiende con varios hilos y SQLite no comparte conexiones entre hilos.

## Decisiones de diseño

| Decisión | Por qué |
|---|---|
| BPE a nivel de byte, propio, 16k tokens | Cubre cualquier texto sin token desconocido. Con 16k, el embedding cuesta 10,5M parámetros; 32k o 50k serían demasiados para un modelo de 56M |
| Pre-tokenización de Llama 3 (tokenizer v2) | Exportable a GGUF con tokenización idéntica en llama.cpp. Cuesta un 1,8 % de compresión en Wikipedia |
| Datos en disco como uint16 + `np.memmap` | Miles de millones de tokens con 16 GB de RAM: solo se leen las ventanas de cada batch |
| Micro-batches de 4.096 tokens + gradient accumulation | Con 16k de vocabulario, los logits de 16k tokens ocupan ~3,5 GiB en fp32. Al pasar de 8 GiB, Windows desborda a RAM sin avisar y la velocidad cae de 10 a 30 veces |
| `torch.compile` | ×1,5–1,6 de throughput, porque fusiona RMSNorm, RoPE, SwiGLU y la loss |
| GQA con `repeat_interleave` y no `enable_gqa` | `enable_gqa` resultó un 20 % más lento en esta GPU: abandona el kernel de flash attention |
| Checkpoints atómicos con RNG | Los entrenamientos de horas se reparten en sesiones: un corte no debe perder el progreso y el resume debe ser idéntico a no haber parado (`tests/test_training.py`) |
| Tokens especiales solo por id | `encode` ignora los especiales por defecto y el chat template los inserta por id. Así el texto de un usuario o de una web no puede inyectar turnos ni llamadas a herramientas |
| API de un modelo y una petición a la vez | Herramienta local de un solo usuario: un `threading.Lock` basta y mantiene la VRAM bajo control |
| Historial en SQLite, guardado por el servidor | Biblioteca estándar y un solo archivo. Guarda el servidor, no el navegador, así que una respuesta detenida o fallida queda registrada igual, y la base de datos es la única fuente de verdad |
| Exportar a GGUF propio (`leon.export`) en vez de pasar por Hugging Face | Leon ya es un Llama con el RoPE de llama.cpp: los pesos se copian tal cual, sin permutar. Se verifica contra los binarios de llama.cpp (tokens, chat template, perplejidad) |
| API solo local | Escucha en `127.0.0.1` y valida la cabecera `Host`, contra el *DNS rebinding* |
| Web separada (React) de la API (Flask) | Se desarrollan por separado (`npm run dev` con proxy). En uso normal, Flask sirve la web compilada desde el mismo puerto |

## Reglas de dependencias

1. **`leon/` no importa nada** de `phases/`, `apps/` ni `tests/`: solo la biblioteca estándar, numpy, torch, regex (clases Unicode del pre-tokenizer) y sus propios módulos. La única excepción son los extras opcionales de `export.py` (`gguf`, `jinja2`), que no importa ningún otro módulo.
2. **Las fases 2 en adelante** importan de `leon/` y nunca entre ellas. Si dos fases necesitan lo mismo, ese código se gradúa a `leon/`.
3. **Las fases 0 y 1** no importan de `leon/`: son prototipos autocontenidos y se pueden leer sin conocer el resto.
4. **`apps/api`** importa de `leon/`. **`apps/web`** solo habla con la API por HTTP.
5. **Ningún script modifica `sys.path`** para encontrar `leon`, que se instala con `pip install -e .`. Las únicas excepciones son los notebooks y `tests/test_api.py`, que cargan código que no es un paquete (`phases/1_gpt`, `apps/api`).

`tests/test_architecture.py` comprueba automáticamente las reglas 1, 2 y 3.

## Cómo encaja cada fase (4–8)

Todas siguen el mismo patrón: un `prepare.py` que deja los datos en `data/<fase>/`, un `train.py` que escribe `status.json` y checkpoints con `leon.run.RunFiles` y `leon.training.save_checkpoint`, y un `control.py` con `start / pause / resume / status / watch` y el panel común (`leon.run`). Pausar y continuar da los mismos pesos que entrenar sin parar.

| Fase | Dónde | Qué añade al producto | Qué reutiliza |
|---|---|---|---|
| 4: preentrenamiento | `phases/4_pretrain/`: `control.py` (`prepare`, `start`, `pause`, `resume`, `status`, `watch`, `extend`) lanza `train.py` como proceso independiente; `extend` continúa a contexto 2048 con `--init-from` y `--rope-theta` | `leon/run.py`, `TokenFile.long_windows`, `leon.training.loss_by_position` | `leon.data`, `leon.model`, `leon.training` |
| 5: SFT conversacional | `phases/5_sft/`: `prepare.py` (diálogos de oasst2, Aya, Dolly, Alpaca y WildChat, codificados con `encode_chat_for_training`, targets a `-100` fuera de las respuestas) y `train.py` sobre `leon.sft.train_sft`. Checkpoints con `stage="chat"` | `leon/sft.py`, `leon.data.ConversationFile`, `leon.run.Control` | `leon.chat`, `leon.training`; la loss de `Leon.forward` ya ignora los targets `-100` |
| 6: herramientas | `phases/6_agent/`: `prepare.py` (diálogos sintéticos con los resultados de las herramientas reales y los documentos de `data/docs/ejemplo/`), `train.py` (`train_sft(..., stage="agent")`), `evaluate.py`. `apps/api/engine.py` ejecuta el bucle con `"leon": {"tools": true}` | llamadas en `leon/chat.py`, `leon/tools.py`, `leon/agent.py`; `tools`/`tool_use`/`tool_result` en `apps/api` | `leon.sft`, `leon.run.Control` |
| 7: DPO | `phases/7_dpo/`: `control.py prepare` genera en la GPU los pares (respuesta humana frente a una muestra de Leon), `train.py` entrena con una referencia congelada precalculada (`reference.pt`) y `evaluate.py` mide bucles, respuestas terminadas y longitud. Hereda el `stage` del modelo de partida | `leon/dpo.py` | `leon.chat`, `leon.training`, `leon.run.Control` |
| 8: razonamiento | `phases/8_reasoning/`: `prepare.py` (aritmética en 5 niveles, con pasos y en versión directa), `train.py` sobre `leon.sft` (`leon_0.1_steps` y el control `leon_0.1_direct`), `grpo.py` (`control.py grpo`) y `evaluate.py`. GRPO y `evaluate.py` no se llegaron a ejecutar con el modelo final: quedan como ejercicio | `leon/grpo.py` | `leon.sft`, `leon.dpo`, `leon.run.Control` |

La web no necesita cambios para un modelo nuevo: `apps/api` lista cualquier `checkpoints/N_*/<run>/best.pt` con N ≥ 3, y la web decide el modo y el interruptor de herramientas por su `stage`.
