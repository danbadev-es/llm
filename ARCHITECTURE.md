# Arquitectura

Mapa técnico del repositorio: qué piezas hay, cómo fluyen los datos entre ellas, qué formatos comparten y qué reglas no deben romperse. Para la motivación y los conceptos de fondo, ver [`docs/fundamentos.md`](docs/fundamentos.md). Para el detalle y los resultados de cada etapa, ver el `README.md` de cada fase.

## Visión general

El repositorio tiene dos mitades con propósitos distintos:

- **El curso** (`phases/`): una carpeta por fase, que se lee en orden. Cada fase contiene sus scripts, un README con resultados medidos y un notebook que reconstruye las piezas paso a paso.
- **El producto** (`leon/` y `apps/`): el código definitivo de Leon 0.1, que se reutiliza, está testeado y es estable.

Cuando una fase produce un componente que usarán fases posteriores o las apps, ese componente se **gradúa** a `leon/`. La fase conserva los scripts que lo entrenan, lo miden y lo explican. Las fases 0 y 1 son prototipos autocontenidos: no dependen de `leon/` y nada depende de ellas.

```mermaid
flowchart LR
    subgraph course["Curso: phases/"]
        P0["0_bigram"]
        P1["1_gpt"]
        P2["2_tokenizer"]
        P3["3_modern"]
        P4["4_pretrain … 8_reasoning<br/>(por venir)"]
    end
    subgraph product["Producto"]
        L["leon/<br/>paquete Python"]
        API["apps/api<br/>Flask"]
        WEB["apps/web<br/>React + Vite"]
    end
    P2 -- "gradúa tokenizer" --> L
    P3 -- "gradúa modelo, datos, entrenamiento" --> L
    P2 & P3 & P4 -- "importan" --> L
    API -- "importa" --> L
    WEB -- "HTTP /api" --> API
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
    C -->|"engine.py"| API2["apps/api"]
    TK --> API2
    API2 -->|"Server-Sent Events"| UI["apps/web"]
```

`data/` y `checkpoints/` no se versionan: todo se reconstruye con los scripts de las fases.

## El paquete `leon/`

| Módulo | Responsabilidad | Depende de |
|---|---|---|
| `paths.py` | Rutas de `data/`, `checkpoints/` y del tokenizer | — |
| `tokenizer.py` | `BPETokenizer`: pre-tokenización por regex, entrenamiento BPE incremental, `encode`/`decode`, tokens especiales, `save`/`load` | — |
| `model.py` | `Leon` y `LeonConfig`: RMSNorm, RoPE, atención con GQA, SwiGLU, `KVCache`, `stream`/`generate` | torch |
| `data.py` | `TokenFile` (`.bin` con `np.memmap`, `get_batch`), `load_splits`, `load_tokenizer` | `paths`, `tokenizer` |
| `training.py` | `lr_at`, `estimate_loss`, `bits_per_byte`, `save_checkpoint`/`load_checkpoint` | `model`, `data` |
| `run.py` | `RunFiles`: ficheros de control de un entrenamiento en segundo plano (`PAUSE`, `status.json`, `train.pid`, `train.log`), `is_process_alive`, `keep_system_awake`. Lo común a los `control.py` de todas las fases: `launch` (proceso sin ventana), `wait_for_pause`, el panel en vivo (`dashboard`, `watch`), que muestra los campos que tenga el `status.json` de cada fase, y `Control`, la línea de comandos `start / pause / resume / status / watch` | — |
| `sft.py` | El bucle de ajuste supervisado de las fases 5, 6 y 8: `add_sft_args`, `micro_batch_plan` (conversaciones empaquetadas), `validation_loss`, `answer` (preguntas fijas) y `train_sft`, con estado, pausa y checkpoints | `chat`, `data`, `model`, `run`, `training` |
| `export.py` | GGUF para llama.cpp, LM Studio y Ollama; `--verify` lo compara con los binarios de llama.cpp | `model`, `data`; extra opcional `gguf`, `jinja2` |
| `chat.py` | `encode_chat`: el chat template | `tokenizer` |

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
| `stage` | *(fase 5+)* `"chat"` en los modelos con SFT. Si falta, la API lo trata como `"base"` |

Se escriben de forma atómica: primero un `.tmp` y luego `os.replace`.

### Chat template

```
<|system|>…<|end|><|user|>…<|end|><|assistant|>…<|end|><|user|>…<|end|><|assistant|>
```

Lo define `leon.chat.encode_chat`, y es la única implementación: la fase 5 lo usará para entrenar y `apps/api` para la inferencia.

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

Compatible con la API de Claude (`apps/api/anthropic_api.py`), para que el SDK oficial `anthropic` hable con Leon cambiando solo la URL base: `POST /v1/messages` (modo chat, con o sin streaming), `POST /v1/messages/count_tokens`, `POST /v1/complete` (modo completar, la antigua Text Completions) y `GET /v1/models`. Lo propio de Leon va en una clave `leon` (estadísticas de generación, metadatos del checkpoint) y en la cabecera `X-Leon-Conversation` (guardar en el historial). Además, el CRUD de `/api/conversations`. La web usa esta misma API (`apps/web/src/api.js` traduce entre ella y sus componentes). Ver [`apps/README.md`](apps/README.md).

### Historial de conversaciones

SQLite en `data/chat.db` (`apps/api/store.py`):

- **Tablas**: `conversations` (id, título, modo, modelo, fechas) y `messages` (rol, contenido, `prompt` en las respuestas del modo completar, `stats` en JSON, `error`).
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

1. **`leon/` no importa nada** de `phases/`, `apps/` ni `tests/`: solo la biblioteca estándar, numpy, torch, regex (clases Unicode del pre-tokenizer) y sus propios módulos.
2. **Las fases 2 en adelante** importan de `leon/` y nunca entre ellas. Si dos fases necesitan lo mismo, ese código se gradúa a `leon/`.
3. **Las fases 0 y 1** no importan de `leon/`: son prototipos autocontenidos y se pueden leer sin conocer el resto.
4. **`apps/api`** importa de `leon/`. **`apps/web`** solo habla con la API por HTTP.
5. **Ningún script modifica `sys.path`** para encontrar `leon`, que se instala con `pip install -e .`. Las únicas excepciones son los notebooks y `tests/test_api.py`, que cargan código que no es un paquete (`phases/1_gpt`, `apps/api`).

`tests/test_architecture.py` comprueba automáticamente las reglas 1 y 3.

## Puntos de extensión (fases 4–6)

| Fase | Dónde | Qué reutiliza |
|---|---|---|
| 4: preentrenamiento | **Implementada**: `phases/4_pretrain/control.py` (prepare, start, pause, resume, status, watch, extend) lanza `train.py` como proceso independiente. Pausar y continuar da exactamente los mismos pesos que entrenar sin parar | `leon.data`, `leon.model`, `leon.training`, `leon.run` |
| 5: SFT conversacional | `phases/5_sft/`: dataset de diálogos codificado con `encode_chat` y targets a `-100` en los turnos no del asistente (`ignore_index` de la loss). Checkpoints con `stage="chat"` | `leon.chat`, `leon.training`; la loss de `Leon.forward` ya ignora los targets `-100` |
| 6: tool calling | `apps/api/engine.py`: detectar `<\|tool_call\|>…<\|/tool_call\|>`, ejecutar la herramienta, insertar `<\|tool_result\|>…<\|/tool_result\|>` y continuar generando. Las herramientas irán en un módulo propio de `leon/` o de `apps/api` | tokens especiales ya reservados |
