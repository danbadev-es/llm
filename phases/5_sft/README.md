# Fase 5: ajuste supervisado (SFT) para conversar

## Objetivo

Convertir el modelo base de la fase 4, que solo continúa texto de Wikipedia, en un **asistente que conversa**: que entienda dónde acaba la pregunta, conteste en el turno del asistente y **sepa cuándo parar**.

Se parte de la extensión a 2048 tokens (`checkpoints/4_pretrain/leon_0.1_ctx2048/best.pt`; si no existe, de `leon_0.1/best.pt`) y se ajusta con ~86.000 conversaciones en español. El resultado (`checkpoints/5_sft/leon_0.1_chat/best.pt`, `stage="chat"`) es el modelo de conversación de la app y el punto de partida de las fases 6–8.

## Archivos

| Archivo | Contenido |
|---|---|
| `prepare.py` | Descarga las cinco fuentes, filtra, deduplica, recorta al contexto y tokeniza (CPU y red) |
| `train.py` | Envoltorio fino: los valores por defecto de la fase (modelo base, `--upsample oasst2=3 aya=2`, preguntas fijas) y una llamada a `leon.sft.train_sft` |
| `control.py` | `start`, `pause`, `resume`, `status`, `watch` en segundo plano, con el panel de siempre |
| [`leon/sft.py`](../../leon/sft.py) | El bucle de SFT, compartido con las fases 6 y 8: empaquetado, *upsampling*, pérdida solo en lo aprendido, evaluación y pausa limpia |
| [`leon/chat.py`](../../leon/chat.py) | El template de chat, la máscara de aprendizaje (`encode_chat_for_training`) y `DEFAULT_SYSTEM` |
| [`leon/data.py`](../../leon/data.py) | `ConversationFile`: lee `{train,val}.npz`, empaqueta conversaciones en bloques y construye los targets |

## Conceptos nuevos

- **Template de chat**: la conversación se escribe con los tokens especiales reservados en la fase 2:

  ```
  <|system|>Eres Leon, un asistente…<|end|><|user|>¿Cuál es la capital de Francia?<|end|><|assistant|>La capital de Francia es París.<|end|>
  ```

  Para generar, el prompt termina en `<|assistant|>` y el modelo escribe hasta `<|end|>`. Un mensaje del usuario que contenga el texto `<|end|><|assistant|>` no puede falsificar un turno: el texto se tokeniza como texto, nunca como token especial.
- ***Loss masking***: el modelo solo aprende los tokens de las respuestas del asistente. El *system prompt*, los turnos del usuario y el propio marcador `<|assistant|>` son contexto: sus targets valen `-100` (`ignore_index`, ver fase 3). Aprende a **responder**, no a imitar preguntas. De 23,1M tokens de train, 16,5M se aprenden.
- **El `<|end|>` aprendido**: el `<|end|>` que cierra cada respuesta sí es un target. Es lo que enseña a parar; el modelo base nunca lo había visto y escribía hasta agotar el presupuesto de tokens.
- **Empaquetado (*packing*)**: las conversaciones, barajadas, se meten enteras en bloques de 2049 tokens hasta llenarlos; ninguna se parte entre dos bloques, y el hueco que sobra es relleno sin target. Con conversaciones de unos cientos de tokens, un bloque por conversación desperdiciaría casi toda la GPU.
- ***Upsampling* de los datos humanos**: oasst2 (×3) y aya (×2) son conversaciones escritas por personas, pero son pocas; alpaca y WildChat son mayoría y están generadas con modelos de OpenAI. Repetir las humanas les da más peso sin tirar nada. Las conversaciones de 2 o más intercambios (~9 %) se repiten ×3 (`--multi-turn`), para que aprenda a seguir una conversación. Se toma el mayor de los dos factores, no el producto.
- **lr 10 veces menor que en el preentrenamiento** (1e-4 → 1e-5, frente a 1e-3): el modelo ya sabe español; se trata de **adaptarlo** al formato, no de reaprender. Con un lr de preentrenamiento, 86.000 conversaciones borrarían parte de lo aprendido en 2B de tokens (*catastrophic forgetting*). Por la misma razón, `dropout` 0,1: los datos son pocos y se ven 3 veces.
- ***System prompt* por defecto**: un 30 % de las conversaciones llevan `DEFAULT_SYSTEM` ("Eres Leon, un asistente que responde en español de forma clara y breve."). Así el modelo funciona con y sin él.

## Datos

`python phases/5_sft/prepare.py` (CPU y red). Solo las filas en español de cada fuente:

| Fuente | Tipo | Licencia | Conversaciones (train) | Tokens (train) |
|---|---|---|---|---|
| `oasst2` (OpenAssistant) | Humanas, varios turnos | Apache-2.0 | 3.570 | 1,2M |
| `aya` (Cohere) | Humanas, pregunta y respuesta | Apache-2.0 | 3.635 | 0,5M |
| `dolly` (curada, es) | Traducidas | CC BY-SA 3.0 | 14.253 | 3,0M |
| `alpaca-spanish` (bertin) | Traducidas, generadas con OpenAI | CC BY 4.0 | 49.341 | 5,1M |
| `WildChat-1M` (Allen AI) | Chats reales con ChatGPT | ODC-BY | 16.121 | 13,3M |
| **Total** | | | **86.088** (+ 4.573 de val) | **23,1M** (16,5M aprendidos) |

- **WildChat**: solo conversaciones en español y no tóxicas. Se salta la conversación entera si alguna respuesta menciona OpenAI o ChatGPT, dice "soy una IA" o lleva código: enseñaría a Leon a ser otro.
- **Identidad**: en oasst2 el asistente se presenta como OpenAssistant; esas menciones se sustituyen por "Leon".
- **Deduplicación**: 77 conversaciones repetidas fuera.
- **Validación**: el 5 %, decidido por el **primer intercambio**. Así todas las ramas de un mismo árbol de oasst2 caen en el mismo lado y la validación no se contamina.
- **Contexto**: las conversaciones más largas que 2048 tokens se recortan a los **intercambios enteros** que caben (2.322 recortadas); si ni el primero cabe, se descartan (890).
- El 91 % de las conversaciones tienen un solo intercambio. Los recuentos completos, en `data/sft/stats.json`.
- Las fuentes generadas con modelos de OpenAI sirven para estudiar, no para un producto comercial.

## Ejecución

```powershell
python phases/5_sft/prepare.py               # una vez (CPU y red)
python phases/5_sft/control.py start         # cuando la GPU esté libre (~2 h)
python phases/5_sft/control.py status        # ¿cómo va?
python phases/5_sft/control.py pause         # guarda y para
python phases/5_sft/control.py resume        # continúa donde se quedó
```

Como en la fase 4, el entrenamiento corre sin ventana y `start`/`resume` abren el panel `watch`; cerrarlo no lo detiene. Lo que va después de `--` se pasa a `train.py` (por ejemplo, `-- --epochs 1` o `-- --upsample`).

Configuración por defecto (`leon/sft.py` y `train.py`):

| Ajuste | Valor |
|---|---|
| Épocas | 3 |
| Batch | 16 bloques de 2049 tokens por paso (2 por micro-batch × `--grad-accum 8`) |
| lr | 1e-4 → 1e-5 (coseno), 50 pasos de warmup |
| Regularización | `dropout` 0,1, `weight-decay` 0,1, `grad-clip` 1,0 |
| *Upsampling* | `oasst2=3 aya=2`, `--multi-turn 3` |
| Evaluación | Cada 100 pasos: loss de las respuestas en val y tres preguntas fijas |

El run de los resultados se lanzó con `train.py` antes de que existiera `control.py`, así que no escribió `status.json`; `control.py status` lo muestra igualmente, leyendo las evaluaciones de `train.log`.

Para probarlo en la app: elegir `5_sft/leon_0.1_chat`.

## Qué observar

1. **Las preguntas fijas en el paso 0**: el modelo base no responde, continúa el texto como si fuera Wikipedia ("Hola, ¿quién eres?" → "o no, conjuntamente, conjuntamente…").
2. **Cuándo aprende el formato**: en pocos cientos de pasos ya contesta en el turno del asistente y termina con `<|end|>`. Lo que tarda más es que la respuesta tenga sentido.
3. **La val loss de las respuestas**: baja muy rápido al principio (el formato) y luego despacio (el contenido).
4. **Los hechos**: el SFT enseña a contestar, no le da conocimientos nuevos. Lo que no sabía en la fase 4 lo sigue inventando, ahora con tono de asistente.

## Ejercicios

- [ ] Comparar en la app el modelo base de la fase 4 y el de la fase 5 con los mismos prompts.
- [ ] Entrenar sin *upsampling* (`-- --upsample --multi-turn 1 --run sin_upsampling`). ¿Cambia el tono, o cómo sigue una conversación de varios turnos?
- [ ] Entrenar 1 época (`-- --epochs 1 --run una_epoca`). ¿Cuánto peor es la val loss? ¿Se nota en el chat?
- [ ] Preparar los datos sin WildChat (`prepare.py --sources oasst2 aya dolly alpaca`). Es más de la mitad de los tokens: ¿responde más corto?
- [ ] Probar la misma pregunta abierta a temperatura 0,2, 0,5 y 0,8.

## Resultados

Entrenado el 2026-11-07 desde `4_pretrain/leon_0.1_ctx2048` (paso 900).

| Métrica | Valor |
|---|---|
| Datos | 86.088 conversaciones de train, 23,1M tokens (16,5M aprendidos), 3 épocas con *upsampling* |
| Pasos | 4.863 de 16 bloques × 2049 tokens (38.908 micro-batches) |
| Tiempo de GPU | **1 h 53 min**, ~21.000 tok/s |
| Val loss de las respuestas | 2,846 (ppl 17,2) → **2,070 (ppl 7,9)** |

| Paso | Val loss | "Hola, ¿quién eres?" |
|---|---|---|
| 0 | 2,846 | "o no, conjuntamente, conjuntamente, conjuntamente…" |
| 100 | 2,421 | "¡No te importa que te importa que te importa…" |
| 300 | 2,312 | "¡Hola! ¿En qué puedo ayudarte hoy?" |
| 4.863 | **2,070** | "¡Hola! ¿En qué puedo ayudarte hoy?" |

- **El formato, aprendido**: "¿Cuál es la capital de Francia?" → "La capital de Francia es París.", y termina solo.
- **El contenido, no tanto**: con frases cortas acierta, pero en preguntas abiertas se va por las ramas e inventa datos con seguridad ("Escribe una frase sobre el mar." → "El mar es como un océano en el tiempo."). "Que es la vida" da un párrafo fluido sin sentido a temperatura 0,8; a 0,5 sale mejor. Por eso la temperatura por defecto de la app es ahora **0,5**.
- **Lo que viene**: las fases 6 (herramientas, para los datos que no sabe) y 7 (DPO, para que se repita menos) atacan justo estos fallos.
