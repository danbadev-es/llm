# Fase 6: herramientas y bucle agéntico

## Objetivo

Que Leon use **tres herramientas** cuando le hacen falta y conteste con su resultado:

| Herramienta | Para qué | Por qué la necesita un modelo de 56M |
|---|---|---|
| `calculadora(expresion)` | Aritmética exacta | Un modelo de lenguaje predice dígitos, no calcula: con 56M falla casi cualquier multiplicación |
| `fecha_hora()` | La fecha y la hora | Ningún modelo puede saberlas: no están en sus pesos |
| `buscar_documentos(consulta)` | Pasajes de los documentos del usuario (`data/docs/`) | Sabe muy pocos hechos y los inventa con seguridad; buscar compensa justo eso (es la idea de RAG) |

Se parte del modelo de chat de la fase 5 (`checkpoints/5_sft/leon_0.1_chat/best.pt`) y se ajusta con diálogos que llaman a herramientas. El resultado (`checkpoints/6_agent/leon_0.1_agent/best.pt`, `stage="agent"`) se usa en la app con el interruptor **Herramientas**.

Metas (de [`docs/fundamentos.md`](../../docs/fundamentos.md)): **JSON válido > 95 %** y **herramienta correcta > 80 %** en preguntas del estilo del entrenamiento.

## Archivos

| Archivo | Contenido |
|---|---|
| `prepare.py` | Genera los diálogos (CPU, ~2 min) y los documentos de ejemplo |
| `train.py` | El ajuste: `leon.sft.train_sft`, el mismo bucle de la fase 5 |
| `control.py` | `start`, `pause`, `resume`, `status`, `watch` en segundo plano, con el panel de siempre |
| `evaluate.py` | JSON válido, herramienta correcta y respuesta correcta en preguntas que no vio |
| [`leon/chat.py`](../../leon/chat.py) | El template con llamadas a herramientas |
| [`leon/tools.py`](../../leon/tools.py) | Las tres herramientas y el *system prompt* que las anuncia |
| [`leon/agent.py`](../../leon/agent.py) | El bucle: generar, parsear la llamada, ejecutar, añadir el resultado, seguir |

## Conceptos nuevos

- **Tool calling**: el modelo no ejecuta nada. Escribe la llamada como texto (JSON entre dos tokens especiales) y se detiene; el *runtime* la ejecuta y le devuelve el resultado:

  ```
  <|assistant|><|tool_call|>{"name": "calculadora", "arguments": {"expresion": "17*24"}}<|/tool_call|>
  <|tool_result|>408<|/tool_result|>17 por 24 son 408.<|end|>
  ```

  Los cuatro tokens se reservaron en la fase 2, así que el vocabulario no cambia.
- **Un agente es un bucle, no un modelo** (`leon.agent.run_agent`): generar hasta `<|/tool_call|>`, ejecutar, añadir el resultado y continuar el mismo turno, con un máximo de 3 llamadas y el presupuesto de tokens.
- **Qué se aprende** (*loss masking*): la llamada y la respuesta, nunca el resultado, que lo escribe el *runtime*. El modelo aprende a **pedir** y a **leer**, no a inventar resultados.
- **Datos sintéticos**: no hay un conjunto en español para estas herramientas, así que `prepare.py` genera los diálogos con plantillas, y **cada resultado lo produce la herramienta real**: el modelo entrena con exactamente lo que verá en uso.
- **Cuándo no llamar**: un 30 % son conversaciones de la fase 5 con el mismo *system prompt*. Sin ellas, el modelo llamaría a una herramienta para todo.
- **Decir "no lo sé"**: en las búsquedas sin resultado, la respuesta correcta es decirlo, no inventar (`no_encontrado`).
- **Evaluación honesta**: los documentos que buscan los diálogos de entrenamiento (`data/agent/docs_pool/`) y los de la evaluación (`data/docs/ejemplo/`) son artículos distintos de la validación de Wikipedia. Así se mide si sabe buscar, no si recuerda.

## Datos

`python phases/6_agent/prepare.py` (CPU, ~2 min). Necesita `data/sft/train.jsonl` (fase 5) y `data/wiki_es/val.jsonl` (fase 4).

| Tipo | Diálogos (train) | Ejemplo |
|---|---|---|
| `calculadora` | 7.289 | "¿Cuánto es el 5% de 7773?" → `5*7773/100` → `388,65` → "El 5% de 7773 es 388,65." |
| `fecha_hora` | 2.727 | "¿En qué mes estamos?" → `fecha_hora` → "domingo, 1 de diciembre de 2024, 15:36" → "Estamos en diciembre." |
| `documentos` | 5.521 | "¿Qué dicen mis documentos sobre La Pietrecita?" → la primera frase del pasaje encontrado |
| `no_encontrado` | 939 | "Consulta mis documentos: Pokhara" → sin resultados → "Tus documentos no dicen nada sobre Pokhara." |
| `varias` | 1.130 | "¿Cuántos años han pasado desde 1531?" → `fecha_hora`, luego `calculadora("2025-1531")` |
| `sin_herramienta` | 7.610 | Conversaciones de la fase 5 sin cálculos ni fechas |
| **Total** | **25.216** (5,3M tokens, 1,8M aprendidos) | + 1.471 de validación y 450 preguntas de evaluación |

Los números se escriben como en español (`3,5`), y la calculadora acepta la coma decimal: el modelo copia, no convierte.

## Ejecución

```powershell
python phases/6_agent/prepare.py              # una vez (CPU)
python phases/6_agent/control.py start        # cuando la GPU esté libre (~15–20 min)
python phases/6_agent/control.py status       # ¿cómo va?
python phases/6_agent/evaluate.py             # al terminar
```

El panel muestra, en cada evaluación, las respuestas a cinco preguntas fijas: una por herramienta, un saludo y una pregunta sin herramientas en el *system prompt*. La llamada se ve en el texto (`<|tool_call|>…<|/tool_call|>`).

Para probarlo en la app: elegir `6_agent/leon_0.1_agent` y activar **Herramientas**. Los documentos están en `data/docs/`: los de ejemplo en `ejemplo/`, y se pueden añadir `.md` o `.txt` propios.

## Qué observar

1. **Las preguntas fijas**: cuándo empieza a llamar a la herramienta correcta, y cuándo deja de llamarla para un saludo.
2. **La calculadora**: copia bien la expresión de la pregunta, pero ¿y en los problemas con palabras?
3. **Los documentos**: ¿la respuesta sale del resultado, o se cuela lo que "sabe" el modelo?
4. **Sin herramientas en el *system prompt***: con las instrucciones de la fase 5, ¿sigue llamando?

## Ejercicios

- [ ] Evaluar el modelo de la fase 5 y el de la fase 6 (`evaluate.py --checkpoint …`) y comparar.
- [ ] Añadir una herramienta nueva (por ejemplo, `convertir_unidades`) a `leon/tools.py` y a `prepare.py`, y reentrenar. ¿Cuántos ejemplos necesita?
- [ ] Entrenar sin los diálogos `sin_herramienta` (`--no-tool 0`). ¿Qué pasa con los saludos?
- [ ] Preguntar en la app por un documento propio en `data/docs/`. ¿Dónde falla la búsqueda BM25?

## Resultados

*Pendiente de ejecutar.*

| Métrica | Valor |
|---|---|
| Val loss (tokens aprendidos), inicio → final | |
| JSON válido / herramienta correcta / respuesta correcta | |
| Fase 5 en la misma evaluación (referencia) | JSON —, herramienta 20 % (solo acierta "sin herramienta"), respuesta 20 % (muestra de 15 preguntas) |
| Horas de GPU | |
