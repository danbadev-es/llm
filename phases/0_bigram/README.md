# Fase 0: bigrama a nivel de carácter

## Objetivo

Construir el pipeline completo de un modelo de lenguaje con el modelo más simple posible, que predice el siguiente carácter usando solo el carácter actual.

Conceptos que cubre:

- **Tokenización**: cada carácter se convierte en un id.
- **Objetivo de entrenamiento**: los targets son los inputs desplazados una posición.
- **Logits → softmax → cross-entropy**: el modelo es literalmente una tabla `V×V` de logits.
- **Optimización**: AdamW sobre esa tabla.
- **Evaluación**: separación train/val y estimación de la loss con varios batches.
- **Sampling**: generación autorregresiva y efecto de la temperatura.

## Ejecución

Desde la raíz del repo:

```powershell
python phases/0_bigram/prepare.py
python phases/0_bigram/bigram.py
python phases/0_bigram/bigram.py --temperature 0.5
python phases/0_bigram/bigram.py --temperature 0
```

### Paso a paso

`walkthrough.ipynb` reconstruye el bigrama celda a celda: tokenización, tabla de conteo, cross-entropy a mano, gradiente y temperatura. Corre en CPU en menos de un minuto; en VS Code, selecciona el kernel de `.venv`.

## Qué observar

1. **Baseline uniforme `ln(V)`**: la loss de un modelo que no sabe nada. En el paso 0 la loss sale algo por encima (≈ +0,3–0,5), porque `nn.Embedding` inicializa los logits con N(0,1) y eso equivale a una distribución "confiadamente aleatoria". Si la diferencia fuera grande, la inicialización estaría mal.
2. **Bigrama por conteo**: la solución cerrada del mismo problema. El modelo entrenado con gradiente debería converger cerca de ese valor. Es una comprobación de que el bucle de entrenamiento es correcto.
3. **Train vs val**: en un bigrama prácticamente no hay overfitting. Hay pocos parámetros y muchos datos.
4. **Muestras**: deberían salir sílabas pronunciables en español, sin palabras coherentes. Ese es el techo de un contexto de un solo carácter, y la razón de ser de la fase 1.

## Ejercicios

- [ ] Explicar por qué la loss inicial es ≈ `ln(V)`. Inicializar la tabla a ceros (`nn.init.zeros_`) y comprobar que el paso 0 da exactamente `ln(V)`.
- [ ] Bajar `--steps` hasta que el modelo no llegue a la loss del conteo. ¿Qué indica eso sobre la convergencia?
- [ ] Comparar `--temperature 0` con `1.0`: ¿por qué greedy entra en bucles?
- [ ] Implementar un trigrama por conteo (contexto de 2 caracteres) y comparar la val loss.

## Resultados

| Métrica | Valor |
|---|---|
| Vocabulario | 92 caracteres |
| Baseline `ln(V)` | 4,5218 |
| Val loss (conteo) | 2,3047 |
| Val loss (entrenado) | 2,3061 |
| Bits/char | 3,327 (3,236 bits/byte) |
| Throughput | 1,39M tok/s (CPU) |
