# Fase 8: razonamiento paso a paso

## Objetivo

Medir, en un terreno acotado (la aritmética), las dos formas de enseñar a razonar que usan los modelos grandes:

1. **Ajuste supervisado con los pasos**: el modelo aprende a escribir la solución línea a línea antes de `Respuesta: N`.
2. **Aprendizaje por refuerzo con recompensa verificable (GRPO)**: el modelo prueba varias soluciones por problema, se premian las que aciertan y aprende de sus propios aciertos.

Y compararlas con un **control**: los mismos problemas respondidos directamente, sin pasos. Con 56M de parámetros Leon no va a razonar en general (ver [`docs/fundamentos.md`](../../docs/fundamentos.md#el-razonamiento)), pero en aritmética el efecto de escribir los pasos se puede medir con exactitud.

## Archivos

| Archivo | Contenido |
|---|---|
| `prepare.py` | Problemas de 5 niveles con su solución paso a paso y la directa (CPU, ~1 min) |
| `train.py` | El ajuste supervisado (`leon.sft`): `leon_0.1_steps` (con pasos) y `leon_0.1_direct` (control) |
| `grpo.py` | El refuerzo desde `leon_0.1_steps` |
| `control.py` | `start`, `grpo`, `pause`, `resume`, `status`, `watch` en segundo plano |
| `evaluate.py` | Aciertos por nivel de cada modelo, con decodificación *greedy* |
| [`leon/grpo.py`](../../leon/grpo.py) | Recompensa, ventajas por grupo y la pérdida de GRPO |

## Conceptos nuevos

- **Escribir es pensar**: cada token cuesta una pasada por las 10 capas. `47 × 36` no cabe en una pasada; dividido en productos de una cifra y sumas de números redondos, sí:

  ```
  ¿Cuánto es 47 × 36?
  47 × 36 = 47 × 30 + 47 × 6
  47 × 30 = 40 × 30 + 7 × 30 = 1200 + 210 = 1410
  47 × 6 = 40 × 6 + 7 × 6 = 240 + 42 = 282
  1410 + 282 = 1692
  Respuesta: 1692
  ```

- **El control**: los mismos 19.000 problemas, respondidos solo con `Respuesta: N`. La diferencia de aciertos entre los dos modelos es lo que valen los pasos.
- **Recompensa verificable**: un programa comprueba la respuesta (`Respuesta: N` correcta = 1; formato correcto con número erróneo = 0,1; nada = 0). No hace falta un modelo que juzgue, ni pares de preferencias.
- **GRPO** (`leon/grpo.py`): 8 intentos por problema; la ventaja de cada uno es su recompensa menos la media de su grupo, dividida por la desviación. Los tokens de los intentos mejores que la media suben de probabilidad y los peores bajan, con una penalización KL hacia el modelo de partida.
- **Grupos que enseñan**: si los 8 intentos aciertan, o fallan todos, la ventaja es 0 y no se aprende nada. Por eso GRPO usa los niveles 2–5 y parte del modelo con pasos, que ya acierta a veces. El panel muestra qué fracción de grupos enseña algo.
- **Nadie le dice cómo pensar**: en el refuerzo, el modelo solo ve qué intentos suyos acertaron. En modelos grandes (DeepSeek-R1) de ahí surgen razonamientos más largos y autocorrecciones; aquí se puede observar si cambia la longitud.

## Datos

`python phases/8_reasoning/prepare.py` (CPU, ~1 min):

| Nivel | Ejemplo | Pasos típicos |
|---|---|---|
| 1 | `78 + 86` | decenas, unidades, suma |
| 2 | `47 × 6` | 1 línea |
| 3 | `47 × 36` | 4 líneas |
| 4 | `(19 + 66) × 3`, `47 × 6 − 38` | 5–6 líneas |
| 5 | "Tengo 5 estanterías con 51 libros cada una y regalo 48 libros. ¿Cuántos libros me quedan?" | 5–6 líneas |

- 4.000 problemas por nivel (95 % train), más 3.000 conversaciones de la fase 5 para que no olvide conversar.
- 5.000 problemas (niveles 2–5) con su respuesta para GRPO.
- 100 por nivel para la evaluación, que nunca aparecen en el entrenamiento.

## Ejecución

Todo usa la GPU salvo `prepare.py`.

```powershell
python phases/8_reasoning/prepare.py
python phases/8_reasoning/control.py start                          # con pasos (~10 min)
python phases/8_reasoning/control.py start --run leon_0.1_direct    # control (~5 min)
python phases/8_reasoning/control.py grpo                           # refuerzo (300 pasos)
python phases/8_reasoning/evaluate.py                               # la tabla de aciertos
```

Parten del modelo de conversación más reciente (fase 7, 6 o 5). `pause`, `status` y `watch` aceptan `--run leon_0.1_direct` o `--run leon_0.1_grpo`; `grpo` continúa solo si estaba en pausa.

## Qué observar

1. **Directo frente a pasos**, por nivel: ¿en qué nivel se separan?
2. **Dónde fallan los pasos**: ¿en un producto de una cifra, en una suma, o al copiar un resultado intermedio?
3. **GRPO**: la recompensa media, la fracción de grupos que enseñan, la longitud de las respuestas y la KL. Si la KL se dispara, el modelo se aleja demasiado de la referencia.

## Ejercicios

- [ ] Entrenar GRPO desde `leon_0.1_direct`. ¿Puede aprender a escribir pasos que nunca vio?
- [ ] Cambiar la bonificación de formato a 0. ¿Qué pasa al principio, cuando casi nada acierta?
- [ ] Añadir un nivel 6 (tres cifras por dos) solo en la evaluación. ¿Generaliza el procedimiento?

## Resultados

Ejecutado el 2026-11-07, desde el modelo de la fase 7. **Se paró tras el ajuste con pasos**, porque era suficiente para el objetivo educativo: el GRPO y `evaluate.py` quedan como ejercicio (el código está probado de punta a punta en CPU).

**Primer intento (4.000 problemas por nivel, 2 épocas)**: el modelo aprendió el **formato** a la perfección, pero no las **operaciones elementales**: "40 × 30 = 1600", "20 + 30 = 40", "140 + 28 = 160". La val loss era baja (0,155) porque casi todos los tokens son la plantilla; basta un dígito mal para fallar el problema. Acertaba el 4,5 % (200 problemas, *greedy*), y en el GRPO ningún grupo de 8 intentos tenía un acierto: sin aciertos no hay nada que reforzar (ventaja 0 en todos los grupos).

**Segundo intento (12.000 problemas por nivel, 4 épocas, 11,7 min)**: val loss 0,0248, y las preguntas fijas, que no están en los datos, salen bien paso a paso:

```
¿Cuánto es (25 + 38) × 4?
Primero el paréntesis: 25 + 38
Decenas: 20 + 30 = 50
Unidades: 5 + 8 = 13
50 + 13 = 63
Después: 63 × 4
63 × 4 = 60 × 4 + 3 × 4 = 240 + 12 = 252
```

```
Tengo 7 cajas con 24 lápices cada una y regalo 15 lápices. ¿Cuántos lápices me quedan?
Lápices en total: 7 × 24
24 × 7 = 20 × 7 + 4 × 7 = 140 + 28 = 168
Me quedan: 168 − 15
168 − 10 = 158
158 − 5 = 151
Respuesta: 151
```

`47 × 36` también acierta sus productos parciales (1410 y 282). El control sin pasos (`leon_0.1_direct`, primer conjunto de datos) responde "Respuesta: 420" a 47 × 36 (son 1692): sin escribir los pasos no tiene dónde hacer la cuenta.

**Coste**: el modelo se especializa. A "Hola, ¿quién eres?" contesta "Como modelo de lenguaje, no tengo la capacidad de conocer el contexto…": 3.000 conversaciones mezcladas no bastan frente a 57.000 problemas. Para conversar, el modelo de la fase 7.

| Modelo | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 | Nivel 5 |
|---|---|---|---|---|---|
| Directo | | | | | |
| Con pasos | | | | | |
| Con pasos + GRPO | | | | | |

*(Pendiente: `python phases/8_reasoning/evaluate.py` y `control.py grpo`.)*
