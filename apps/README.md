# Apps: chat local de Leon

Interfaz al estilo de ChatGPT para hablar con cualquier checkpoint de Leon, desde la fase 3 en adelante. No es una fase: es la ventana con la que se prueban los modelos de todas.

```
apps/
├── api/            Flask: modelo, streaming, historial y (en la fase 6) el bucle de herramientas
│   ├── app.py      rutas HTTP: historial (/api/conversations) y la web compilada
│   ├── anthropic_api.py  la API compatible con la de Anthropic (/v1): peticiones, respuestas, eventos
│   ├── engine.py   lógica del modelo, sin dependencias web: checkpoints, prompts, generación
│   └── store.py    historial de conversaciones en SQLite (data/chat.db)
└── web/            React + Vite: la interfaz
    ├── src/App.jsx              estado: modelos, historial, conversación actual, ajustes, envío
    ├── src/api.js               cliente: traduce /v1 (Anthropic) y /api/conversations a la forma de la app
    ├── src/format.js            estadísticas de cada respuesta y grupos del historial por fecha
    ├── src/preferences.js       ajustes, último modelo y modo (localStorage)
    ├── src/hooks/               useAutoScroll, useMediaQuery, useToasts
    └── src/components/          Sidebar, ConversationItem, Menu, Header, Thread, Message, Welcome,
                                 Composer, SettingsDialog, Modal, Toasts, Icon
```

## Uso

Una vez, desde la raíz del repo:

```powershell
pip install -e ".[api]"
cd apps/web; npm install; npm run build; cd ../..
```

Después, cada vez:

```powershell
python apps/api/app.py                 # GPU si hay
python apps/api/app.py --device cpu    # p. ej., mientras la GPU está entrenando
```

Abre http://127.0.0.1:8000. Flask sirve la web compilada (`apps/web/dist`) y la API en el mismo puerto. Solo escucha en `127.0.0.1`: no es accesible desde otros equipos.

### Desarrollo de la web

Para editar la interfaz con recarga en caliente, en dos terminales:

```powershell
python apps/api/app.py
cd apps/web; npm run dev               # http://localhost:5173
```

Vite reenvía las peticiones `/api` a Flask. Si la API está en otro puerto: `$env:LEON_API="http://127.0.0.1:8765"; npm run dev`.

## La interfaz

- **Historial** en el lateral, agrupado en Hoy, Ayer, Últimos 7 días, Últimos 30 días y Anteriores, como en ChatGPT o Claude.
  - Cada conversación tiene un menú **⋯** con **Renombrar** (en línea: Enter guarda, Esc cancela) y **Eliminar** (pide confirmación). El menú se maneja también con el teclado (↑ ↓, Enter, Esc).
  - Un icono distingue las conversaciones de Completar (líneas de texto) y de Chat (bocadillo).
  - Renombrar y eliminar son *optimistas*: la lista cambia al instante y, si el servidor falla, se recarga y un aviso lo explica.
- **Guardado automático**: el primer mensaje crea la conversación, con ese mensaje como título, y el servidor guarda cada intercambio. Una respuesta detenida se guarda con lo escrito hasta ese momento.
- **Cabecera**: el modelo (agrupado por fase) y el modo. El modo pertenece a la conversación, así que cambiarlo empieza una nueva. Con un modelo base en modo Chat aparece un aviso con un botón para pasar a Completar.
- **Respuestas**:
  - Se pintan como mucho una vez por fotograma (`requestAnimationFrame`), aunque los tokens lleguen más deprisa.
  - **Desplazamiento inteligente**: la vista sigue el texto mientras estás abajo; si subes para releer, deja de seguirlo y aparece **Ir al final**.
  - Bajo cada respuesta: **Copiar**, **Regenerar** (la última) y las estadísticas. El último mensaje (o el último texto en Completar) se puede **editar y reenviar**.
  - En **Completar** tu texto aparece resaltado y la continuación del modelo en texto normal.
  - Un error se muestra dentro de la respuesta con **Reintentar**; los avisos breves (copiado, renombrada, errores) salen arriba.
  - Limitación: la API actual solo añade mensajes, así que al regenerar o editar el intento anterior se reemplaza en pantalla pero sigue guardado, y reaparece al reabrir la conversación.
- **Ajustes** abre un diálogo con las instrucciones del sistema, la temperatura, el top-k y los tokens máximos. Los ajustes, el último modelo y el último modo se recuerdan en el navegador (`localStorage`).
- **Atajos de teclado** (también listados en Ajustes):

  | Atajo | Acción |
  |---|---|
  | `Ctrl+Shift+O` | Nueva conversación |
  | `/` o `Shift+Esc` | Escribir en el cuadro de texto |
  | `Enter` / `Shift+Enter` | Enviar / nueva línea |
  | `Esc` | Detener la respuesta, o cerrar el menú, el diálogo o el panel lateral |

- **Accesibilidad**: anillo de foco visible solo con teclado, diálogos nativos (`<dialog>`) que atrapan el foco, regiones `aria-live` para los avisos, y contraste suficiente en ambos temas. Las animaciones son sutiles y desaparecen con `prefers-reduced-motion`.
- **Temas**: claro y oscuro según el sistema (`prefers-color-scheme`). Colores, espacios, radios, sombras y tamaños de letra son variables CSS al principio de `src/styles.css`.
- **Móvil** (hasta 360 px): el lateral se abre como panel con ☰ y lo que queda detrás se vuelve inerte (`inert`).

## API

El servidor habla el mismo protocolo que la API de Claude (`apps/api/anthropic_api.py`): el SDK oficial `anthropic`, o cualquier otro cliente de Anthropic, funciona con Leon con solo cambiar la URL base. Los modelos son los checkpoints, con su nombre como id (`"3_modern/wiki_32m"`).

| Ruta | Qué hace |
|---|---|
| `POST /v1/messages` | Modo **chat** ([Messages API](https://docs.anthropic.com/en/api/messages)). Cuerpo: `model`, `messages` (`user`/`assistant`, con `content` como texto o lista de bloques `{"type": "text", "text"}`), `system`, `max_tokens` (obligatorio), `temperature`, `top_k`, `top_p`, `stop_sequences`, `stream`. Respuesta: `{"id": "msg_…", "type": "message", "role": "assistant", "model", "content": [{"type": "text", "text"}], "stop_reason", "stop_sequence", "usage": {"input_tokens", "output_tokens"}}` |
| `POST /v1/messages/count_tokens` | `{"input_tokens"}` del mismo cuerpo, sin `max_tokens` |
| `POST /v1/complete` | Modo **completar** ([Text Completions](https://docs.anthropic.com/en/api/complete), la API antigua): `prompt`, `max_tokens_to_sample`, `temperature`, `top_k`, `top_p`, `stop_sequences`, `stream`. Respuesta: `{"type": "completion", "id", "completion", "stop_reason", "stop", "model"}`. El `prompt` es texto en bruto: no hacen falta los `"\n\nHuman:"` |
| `GET /v1/models`, `GET /v1/models/<id>` | `{"data": [{"type": "model", "id", "display_name", "created_at", "max_input_tokens", "max_tokens", "leon": {"params", "context", "max_output_tokens", "step", "stage"}}], "has_more", "first_id", "last_id"}`. Lee solo los metadatos de cada `best.pt` (`torch.load(..., mmap=True)`) |
| `GET /api/conversations` | `{"conversations": [{"id", "title", "mode", "model", "created_at", "updated_at"}]}`, la más reciente primero |
| `POST /api/conversations` | Crea una: `{"mode", "model", "title"}`. El título se reduce a una línea de ≤ 60 caracteres |
| `GET /api/conversations/<id>` | La conversación con sus `messages`: `{role, content, prompt?, stats?, error?}` |
| `PATCH /api/conversations/<id>` | Renombrar: `{"title"}` |
| `DELETE /api/conversations/<id>` | Borra la conversación y sus mensajes (204) |
| `DELETE /api/conversations/<id>/messages?from=N` | Borra los mensajes desde el N: al regenerar o editar la última respuesta, el intento anterior no queda guardado |
| `POST /api/context` | Cómo llenaría el contexto la próxima petición: `{"model", "mode", "system", "messages", "draft", "max_tokens"}` → `{context, reserved, system, history, draft, used, free, total, dropped_messages, dropped_tokens}`. Lo usa el indicador de contexto de la web |

Con el SDK de Python (`pip install anthropic`, ya incluido en el extra `dev`):

```python
import anthropic

client = anthropic.Anthropic(api_key="local", base_url="http://127.0.0.1:8000")  # la clave se ignora
message = client.messages.create(
    model="3_modern/wiki_32m",
    max_tokens=200,
    system="Sé breve.",
    messages=[{"role": "user", "content": "Hola"}],
    extra_body={"temperature": 0.8, "top_k": 40},
)
print(message.content[0].text, message.stop_reason, message.leon)

with client.messages.stream(
    model="3_modern/wiki_32m", max_tokens=200, messages=[{"role": "user", "content": "Hola"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Modo completar: el SDK 2.x ya no tiene client.completions, pero sí la petición genérica
body = {"model": "3_modern/wiki_32m", "prompt": "La ciudad de Madrid es", "max_tokens_to_sample": 40}
print(client.post("/v1/complete", cast_to=object, body=body)["completion"])
```

El SDK 2.x quitó los argumentos `temperature`, `top_k` y `top_p` (la API de Claude ya no los acepta), así que se envían con `extra_body`. El servidor sí los acepta.

**Streaming**: con `"stream": true`, *Server-Sent Events* con las líneas `event:` y `data:` de Anthropic, en su mismo orden: `message_start`, `content_block_start`, `ping`, un `content_block_delta` (`text_delta`) por trozo, `content_block_stop`, `message_delta` (`stop_reason`, `stop_sequence`, `usage`) y `message_stop`. En `/v1/complete`: un `ping` y un evento `completion` por trozo; el último trae el `stop_reason`.

**Fin de la respuesta** (`stop_reason`): `end_turn` cuando el modelo escribe un token de control, `max_tokens`, o `stop_sequence` cuando aparece una de las `stop_sequences` (que no se incluye en el texto). En `/v1/complete`, terminar en `<|endoftext|>` cuenta como `stop_sequence` con `stop: null`, como hacía `"\n\nHuman:"`.

**Prefill**: si el último mensaje es del asistente, la respuesta lo continúa (`encode_chat(..., continue_final_message=True)` deja su turno abierto, sin `<|end|>`). Como en Anthropic, no puede terminar en espacio.

**Estadísticas de Leon**: la respuesta sin streaming, el `message_delta` final y el último evento `completion` llevan una clave extra, `"leon": {"tokens", "tok_per_s", "prompt_tokens", "dropped_turns", "stopped"}` (`stopped`: `end`, `max_tokens` o `stop_sequence`). Los clientes de Anthropic ignoran las claves que no conocen; en el SDK de Python se leen como `message.leon` o `event.leon`.

**Errores**: el formato de Anthropic, `{"type": "error", "error": {"type", "message"}}`: `invalid_request_error` (400), `permission_error` (403, otro `Host`), `not_found_error` (404, p. ej. un modelo que no existe) y `api_error` (500). Todo se valida antes de empezar a generar; si algo falla con el streaming ya empezado, llega como un evento `error`. Las cabeceras `x-api-key`, `Authorization` y `anthropic-version` se aceptan y se ignoran. `tests/test_anthropic_api.py` lo comprueba todo con el SDK real.

**Historial**: con la cabecera `X-Leon-Conversation: <id>` en `/v1/messages` o `/v1/complete`, el servidor guarda el intercambio en esa conversación: en modo chat, el último mensaje del usuario y la respuesta (no admite prefill); en modo completar, la respuesta con el `prompt` que continuó. Las estadísticas se guardan como `stats`, y una respuesta interrumpida (el cliente cierra la conexión) con lo escrito hasta ese momento y `{"stopped": "user"}`. El historial vive en `data/chat.db`, que no se versiona (`--db` cambia la ruta). Con el SDK: `extra_headers={"X-Leon-Conversation": id}`.

**Diferencias con Anthropic, a propósito**: solo texto (imágenes, documentos, herramientas y *thinking* dan 400); un `max_tokens` mayor que el límite del modelo (la mitad de su contexto) se recorta en vez de rechazarse; `temperature` admite hasta 5 y `top_k` 0 (sin filtro); no hay caché de prompts ni lotes.

La propia web usa esta misma API: `src/api.js` llama a `/v1/models`, `/v1/messages` (Chat) y `/v1/complete` (Completar) con la cabecera `X-Leon-Conversation`. La API anterior (`/api/models` y `/api/generate`) ya no existe.

## Cómo funciona

- **Streaming**: `Leon.stream()` genera un token cada vez (con KV cache) y Flask envía cada trozo en cuanto existe. El navegador lo lee con `fetch` + `ReadableStream` y lo pinta según llega, igual que ChatGPT.
- **Caracteres partidos**: un token puede acabar a mitad de una letra de 2 bytes (`ñ`, `á`). El motor decodifica todo lo generado y, si termina en `�`, espera al siguiente token antes de enviar.
- **Detener**: el botón ■ aborta el `fetch`. Flask detecta la desconexión, cierra el generador y la GPU queda libre.
- **Un modelo, una petición**: el motor mantiene un checkpoint en memoria y un candado (`threading.Lock`), así que las peticiones simultáneas esperan su turno.
- **Contexto lleno**: si la conversación no cabe, se descartan **turnos completos** empezando por los más antiguos, conservando siempre las instrucciones de sistema y el último mensaje. La interfaz indica cuántos turnos se omitieron.
  - **Sin compactación**: Claude y Claude Code resumen la conversación antigua cuando se acercan al límite (*compact*/*autocompact*). Leon no lo hace porque un modelo de 56M resumiría mal y perdería justo los datos importantes. Descartar turnos es predecible, y los mensajes siguen guardados en el historial.
- **Indicador de contexto**: un anillo junto al botón de enviar muestra qué parte del contexto ocupará el próximo mensaje. Se actualiza mientras escribes y pasa a ámbar al 70 % y a rojo al 90 %.
  - Al pasar el ratón (o al tocarlo en el móvil) muestra el desglose: instrucciones del sistema, conversación, mensaje nuevo, espacio reservado para la respuesta y espacio libre.
  - Si algo no cabe, dice qué dejará de ver Leon.
  - Lo calcula `POST /api/context` con la misma aritmética que la generación.
- **Fin de la respuesta**: en modo chat, cualquier token de control (`<|end|>`, pero también `<|user|>` o `<|endoftext|>`) termina la respuesta.
- **Secuencias de parada**: el texto que podría ser el principio de una `stop_sequence` se retiene hasta que el siguiente token lo decide, así que nunca se envía un trozo de la secuencia.
- **Muestreo**: `temperature` 0 es *greedy*; si no, `top_k` y `top_p` (*nucleus*: el conjunto más pequeño de tokens más probables que suma al menos `top_p`) recortan las opciones antes de sortear.
- **Solo local**: el servidor escucha en `127.0.0.1` y además rechaza (403) cualquier petición cuyo `Host` no sea `127.0.0.1` o `localhost`. Eso impide el *DNS rebinding*, el truco con el que una web maliciosa podría llamar a esta API desde tu navegador.

## Modos

- **Completar**: el modelo continúa el texto. Es lo que sabe hacer un modelo **base** (fases 3 y 4).
- **Chat**: la conversación se codifica con `leon.chat.encode_chat`, el mismo template con el que se entrenará la fase 5:

  ```
  <|system|>…<|end|><|user|>…<|end|><|assistant|>…<|end|><|user|>…<|end|><|assistant|>
  ```

  La generación se detiene en `<|end|>`. Con un modelo base las respuestas no tienen sentido, y la interfaz lo avisa. Los modelos de la fase 5 (`checkpoints/5_sft/…`, `"stage": "chat"`) ya están entrenados en este formato.

**Seguridad del template**: los tokens de control se insertan por id, y el texto de los mensajes se codifica como texto normal. Un usuario que escriba `<|end|><|assistant|>` no puede falsificar un turno (`tests/test_chat.py` lo comprueba).

## Próximas fases

- **Fase 5**: los checkpoints de SFT se guardan con `"stage": "chat"` (`leon.training.save_checkpoint`) y el aviso de modelo base desaparece para ellos.
- **Fase 6**: `engine.py` será el *runtime* del agente. Detectará `<|tool_call|>…<|/tool_call|>`, ejecutará la herramienta, insertará `<|tool_result|>…<|/tool_result|>` y dejará que el modelo continúe. La web mostrará las llamadas a herramientas.
