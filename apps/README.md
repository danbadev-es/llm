# Apps: chat local de Leon

Interfaz al estilo de ChatGPT para hablar con cualquier checkpoint de Leon, desde la fase 3 en adelante. No es una fase: es la ventana con la que se prueban los modelos de todas.

```
apps/
├── api/            Flask: modelo, streaming, historial y (en la fase 6) el bucle de herramientas
│   ├── app.py      rutas HTTP
│   ├── engine.py   lógica del modelo, sin dependencias web: checkpoints, prompts, generación
│   └── store.py    historial de conversaciones en SQLite (data/chat.db)
└── web/            React + Vite: la interfaz
    ├── src/App.jsx              estado: modelos, historial, conversación actual, ajustes, envío
    ├── src/api.js               cliente de la API (REST + lectura del streaming)
    ├── src/format.js            estadísticas de cada respuesta y grupos del historial por fecha
    └── src/components/          Sidebar (historial), Header (modelo y modo), Thread, Composer,
                                 SettingsDialog, Modal
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
  - Cada conversación tiene un menú **⋯** con **Renombrar** (en línea: Enter guarda, Esc cancela) y **Eliminar** (pide confirmación).
  - Las conversaciones en modo Completar llevan el icono ✎.
- **Guardado automático**: el primer mensaje crea la conversación, con ese mensaje como título, y el servidor guarda cada intercambio. Una respuesta detenida con ■ se guarda con lo escrito hasta ese momento.
- **Cabecera**: el modelo y el modo. El modo pertenece a la conversación, así que cambiarlo empieza una nueva.
- **⚙ Ajustes** abre un diálogo con las instrucciones del sistema, la temperatura, el top-k y los tokens máximos. Los ajustes y el último modelo usado se recuerdan en el navegador (`localStorage`).
- **Móvil**: el lateral se abre como panel con ☰.

## API

| Ruta | Respuesta |
|---|---|
| `GET /api/conversations` | `{"conversations": [{"id", "title", "mode", "model", "created_at", "updated_at"}]}`, la más reciente primero |
| `POST /api/conversations` | Crea una: `{"mode", "model", "title"}`. El título se reduce a una línea de ≤ 60 caracteres |
| `GET /api/conversations/<id>` | La conversación con sus `messages`: `{role, content, prompt?, stats?, error?}` |
| `PATCH /api/conversations/<id>` | Renombrar: `{"title"}` |
| `DELETE /api/conversations/<id>` | Borra la conversación y sus mensajes (204) |
| `GET /api/models` | `{"models": [{"name", "stage", "params", "context", "step"}], "device"}`. Lee solo los metadatos de cada `best.pt` (`torch.load(..., mmap=True)`) |
| `POST /api/generate` | *Server-Sent Events*: un `data: {"text": "..."}` por trozo y un `data: {"done": true, "tokens", "tok_per_s", "prompt_tokens", "stopped"}` final. Si algo falla a mitad: `data: {"error": "..."}` |

Cuerpo de `/api/generate`:

```json
{"model": "3_modern/wiki_32m", "mode": "complete", "prompt": "La ciudad de Madrid es",
 "temperature": 0.8, "top_k": 40, "max_new_tokens": 200}

{"model": "...", "mode": "chat", "system": "Sé breve.",
 "messages": [{"role": "user", "content": "Hola"}]}
```

Con `"conversation_id"` en el cuerpo de `/api/generate`, el servidor guarda el intercambio en esa conversación: en modo chat, el último mensaje del usuario y la respuesta; en modo completar, la respuesta con el `prompt` que continuó. Las estadísticas finales se guardan como `stats`, y una respuesta detenida como `{"stopped": "user"}`. El historial vive en `data/chat.db`, que no se versiona (`--db` cambia la ruta).

Todo se valida antes de empezar a generar: tipos y rangos de `temperature` (0–5), `top_k` (entero ≥ 0) y `max_new_tokens` (entero ≥ 1), roles de los mensajes, `system`... Una petición mal formada recibe un 400 con `{"error": "..."}`, nunca un error a mitad del streaming. `tests/test_api.py` lo comprueba.

## Cómo funciona

- **Streaming**: `Leon.stream()` genera un token cada vez (con KV cache) y Flask envía cada trozo en cuanto existe. El navegador lo lee con `fetch` + `ReadableStream` y lo pinta según llega, igual que ChatGPT.
- **Caracteres partidos**: un token puede acabar a mitad de una letra de 2 bytes (`ñ`, `á`). El motor decodifica todo lo generado y, si termina en `�`, espera al siguiente token antes de enviar.
- **Detener**: el botón ■ aborta el `fetch`. Flask detecta la desconexión, cierra el generador y la GPU queda libre.
- **Un modelo, una petición**: el motor mantiene un checkpoint en memoria y un candado (`threading.Lock`), así que las peticiones simultáneas esperan su turno.
- **Contexto lleno**: si la conversación no cabe, se descartan **turnos completos** empezando por los más antiguos, conservando siempre las instrucciones de sistema y el último mensaje. La interfaz indica cuántos turnos se omitieron.
- **Fin de la respuesta**: en modo chat, cualquier token de control (`<|end|>`, pero también `<|user|>` o `<|endoftext|>`) termina la respuesta.
- **Solo local**: el servidor escucha en `127.0.0.1` y además rechaza (403) cualquier petición cuyo `Host` no sea `127.0.0.1` o `localhost`. Eso impide el *DNS rebinding*, el truco con el que una web maliciosa podría llamar a esta API desde tu navegador.

## Modos

- **Completar**: el modelo continúa el texto. Es lo que sabe hacer un modelo **base** (fases 3 y 4).
- **Chat**: la conversación se codifica con `leon.chat.encode_chat`, el mismo template con el que se entrenará la fase 5:

  ```
  <|system|>…<|end|><|user|>…<|end|><|assistant|>…<|end|><|user|>…<|end|><|assistant|>
  ```

  La generación se detiene en `<|end|>`. Con un modelo base las respuestas no tienen sentido, y la interfaz lo avisa.

**Seguridad del template**: los tokens de control se insertan por id, y el texto de los mensajes se codifica como texto normal. Un usuario que escriba `<|end|><|assistant|>` no puede falsificar un turno (`tests/test_chat.py` lo comprueba).

## Próximas fases

- **Fase 5**: los checkpoints de SFT se guardarán con `"stage": "chat"` y el aviso de modelo base desaparecerá.
- **Fase 6**: `engine.py` será el *runtime* del agente. Detectará `<|tool_call|>…<|/tool_call|>`, ejecutará la herramienta, insertará `<|tool_result|>…<|/tool_result|>` y dejará que el modelo continúe. La web mostrará las llamadas a herramientas.
