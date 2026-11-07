# Fase 7: alineamiento por preferencias (DPO)

## Objetivo

Que Leon prefiera las respuestas buenas a sus propios fallos típicos: repetirse, irse por las ramas, inventar. Para eso se le enseñan **pares**: para una misma pregunta, una respuesta mejor (*chosen*) y una peor (*rejected*), y aprende a hacer más probable la primera **en relación con** la segunda.

- **Buena**: la respuesta humana del conjunto de la fase 5 (oasst2, aya, dolly).
- **Mala**: una respuesta que genera el propio Leon a la misma pregunta. Así el entrenamiento ataca justo los errores que comete este modelo, no los de otro.

El resultado (`checkpoints/7_dpo/leon_0.1_dpo/best.pt`) se usa en la app como cualquier modelo de conversación.

## Archivos

| Archivo | Contenido |
|---|---|
| `prepare.py` | Genera los pares: muestrea respuestas de Leon (GPU, ~30 min) y las empareja con las humanas |
| `train.py` | El entrenamiento DPO, con referencia congelada |
| `control.py` | `prepare`, `start`, `pause`, `resume`, `status`, `watch` en segundo plano |
| `evaluate.py` | Antes y después: cuántas respuestas se repiten, cuántas terminan solas, longitud |
| [`leon/dpo.py`](../../leon/dpo.py) | La pérdida de DPO y las log-probabilidades de una respuesta |

## Conceptos nuevos

- **RLHF sin RL**: el método clásico (InstructGPT) entrena un modelo de recompensa con los pares y luego optimiza el modelo con PPO, muestreando durante el entrenamiento. **DPO** (Rafailov et al., 2023) demuestra que se puede optimizar la misma preferencia **directamente sobre los pares**, con una pérdida de clasificación:

  ```
  loss = −log σ( β · [ (log π(buena) − log π_ref(buena)) − (log π(mala) − log π_ref(mala)) ] )
  ```

  Sin modelo de recompensa ni muestreo durante el entrenamiento: dos pasadas por par.
- **La referencia**: una copia congelada del modelo de partida. La pérdida no premia que la buena sea probable en absoluto, sino que **gane más** que la mala respecto a la referencia. Así el modelo no se aleja mucho de lo que ya sabía. Como no cambia, sus log-probabilidades se calculan una vez al principio (`reference.pt`) y no ocupa VRAM durante el entrenamiento.
- **β**: cuánto puede alejarse de la referencia. Con β = 0,1, una diferencia de 10 nats en log-probabilidad equivale a 1 de "recompensa".
- **La recompensa implícita**: β · (log π − log π_ref) de cada respuesta. Es normal que **bajen las dos**: DPO solo necesita que crezca la diferencia.
- **Arranque exacto en log 2**: en el paso 0 el modelo y la referencia son iguales, así que las dos recompensas son 0 y la pérdida es −log σ(0) = log 2 = 0,693. Si no empieza ahí, algo está mal (por ejemplo, *dropout* activo).
- **Datos *on-policy***: las respuestas malas salen del mismo modelo que se entrena, con temperatura 0,8, como las daría en uso. Cuatro muestras por pregunta van en un solo batch, que cuesta casi lo mismo que una.

## Ejecución

Las dos partes usan la GPU: lanzarlas solo cuando esté libre.

```powershell
python phases/7_dpo/control.py prepare                # pares: ~3.000 preguntas x 4 muestras (~30 min)
python phases/7_dpo/control.py status --run prepare   # ¿cómo va?
python phases/7_dpo/control.py start                  # entrenamiento DPO
python phases/7_dpo/evaluate.py                       # antes y después
```

`prepare` se puede pausar (`pause --run prepare`) y volver a lanzar: retoma donde lo dejó, porque cada par se escribe en `data/dpo/pairs.jsonl` según se genera. Al terminar, reparte los pares en `data/dpo/{train,val}.jsonl` (5 % de validación, por pregunta).

Parte del modelo de la fase 6 si existe; si no, del de la fase 5 (`--base` y `--policy` lo cambian).

## Qué observar

1. **La pérdida** baja desde 0,693, y la **precisión de la recompensa** (*reward accuracy*: pares en los que gana la buena) sube desde 0,5.
2. **El margen** entre recompensas crece. Si crece demasiado rápido, el modelo puede estar aprendiendo un atajo, como que la buena es la más corta o la más larga.
3. **Las preguntas fijas**: ¿se repite menos? ¿termina antes?
4. **`evaluate.py`**: el porcentaje de respuestas en bucle debería bajar sin que las respuestas se vuelvan absurdamente cortas.

## Ejercicios

- [ ] Entrenar con β = 0,05 y β = 0,5. ¿Qué pasa con la longitud de las respuestas?
- [ ] Usar como mala una respuesta a otra pregunta en vez de la de Leon. ¿Aprende lo mismo?
- [ ] Comparar la longitud media de buenas y malas en `data/dpo/train.jsonl`. ¿Hay un atajo que DPO pueda aprovechar?

## Resultados

*Pendiente de ejecutar.* La cadena completa se probó en CPU con 4 preguntas: `prepare`, 2 pasos de entrenamiento (la pérdida empieza en 0,6931) y `evaluate.py`.

| Métrica | Valor |
|---|---|
| Pares (train / val) | |
| Pérdida DPO y precisión de la recompensa en val, inicio → final | 0,693 / 0,5 → |
| Respuestas en bucle, antes → después | |
| Respuestas que terminan solas, antes → después | |
| Horas de GPU (pares + entrenamiento) | |
