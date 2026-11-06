# Fase 1: GPT mínimo a nivel de carácter

## Objetivo

Sustituir la tabla del bigrama por un transformer decoder-only. El modelo pasa de ver **1 carácter** de contexto a ver **256**, y aprende por sí mismo qué partes del contexto importan. Se mantienen el corpus, el split 90/10 y las métricas de la fase 0, para que la comparación sea directa.

## Archivos

| Archivo | Contenido |
|---|---|
| `model.py` | `CausalSelfAttention` (implementación explícita y opción SDPA), `MLP`, `Block`, `GPT` |
| `data.py` | Tokenizer de caracteres, split train/val y batching en la propia GPU |
| `train.py` | Bucle de entrenamiento: AdamW, warmup + cosine, bf16, gradient clipping, gradient accumulation, checkpoints y resume |
| `sample.py` | Generación desde un checkpoint |
| `test_model.py` | Comprobaciones: causalidad, equivalencia con SDPA y loss inicial ≈ `ln(V)` |

## Conceptos nuevos respecto a la fase 0

- **Self-attention**: `softmax(QKᵀ/√d) · V`. Cada posición pondera las anteriores según su relevancia.
- **Máscara causal**: la matriz triangular inferior impide ver el futuro. `test_causality` lo verifica.
- **Multi-head**: 6 cabezas de 64 dimensiones en lugar de una sola de 384.
- **Pre-norm + residuales**: `x + f(LayerNorm(x))`. Es lo que permite apilar capas.
- **Position embeddings aprendidos**: la atención no conoce el orden por sí misma.
- **Weight tying**: el embedding de entrada y la proyección de salida comparten matriz.
- **Init escalada** de las proyecciones residuales (`0.02/√(2·n_layer)`), para que la varianza no crezca con la profundidad.
- **Learning rate schedule**: warmup lineal y luego decaimiento coseno.
- **Gradient clipping**: limita la norma del gradiente y evita saltos destructivos.
- **Weight decay** solo sobre matrices, no sobre biases ni normas.
- **bf16 autocast**: la mitad de memoria y más velocidad en la RTX 5060.
- **Dropout**: regularización, imprescindible aquí porque el corpus es pequeño.
- **Top-k sampling**: descarta los tokens improbables antes de muestrear.

## Ejecución

Desde la raíz del repo:

```powershell
python phases/1_gpt/test_model.py
python phases/1_gpt/train.py
python phases/1_gpt/sample.py --prompt "Dijo don Quijote"
```

La configuración por defecto (6 capas, 6 cabezas, 384 dimensiones, contexto 256) tiene ~10,7M de parámetros y cabe de sobra en 8 GB. Si te quedas sin VRAM, reduce `--batch-size` y compensa con `--grad-accum`.

Si se interrumpe, se continúa con `--resume`.

### Paso a paso

`walkthrough.ipynb` construye la atención desde cero con matrices diminutas (Q, K, V, máscara, √d, multi-head), cuenta los parámetros del modelo real, entrena un GPT de ~110k parámetros en CPU y dibuja qué mira cada cabeza. Corre en CPU en menos de un minuto; en VS Code, selecciona el kernel de `.venv`.

## Qué observar

1. **Comparación con la fase 0**: la val loss en bits/char debería bajar claramente respecto al bigrama. Esa diferencia es lo que aporta el contexto.
2. **Overfitting**: con ~10M de parámetros y ~2M de caracteres, en algún momento la loss de train seguirá bajando y la de val se estancará o subirá. `best.pt` guarda el mejor punto de val, no el último.
3. **Muestras**: palabras reales, puntuación y diálogos con el estilo del Quijote, pero sin coherencia a largo plazo.
4. **Throughput**: anota los tok/s con y sin `--sdpa`. La diferencia mide cuánto aporta un kernel fusionado.

## Ejercicios

- [ ] Entrenar con `--dropout 0` y comparar la separación entre train y val.
- [ ] Barrer `--n-layer 2/4/6/8` y anotar la val loss. Es una primera scaling law casera.
- [ ] Probar `--lr 3e-3` y `--lr 3e-4`. ¿Cuál diverge y cuál converge lento?
- [ ] Quitar la máscara causal y ver qué pasa con la loss de train y con las muestras. Explicar por qué.
- [ ] Medir el tiempo y la VRAM con `--block-size 128/256/512`. ¿Cómo escala la atención con la longitud?

## Resultados

Configuración por defecto (5000 pasos, 16.384 tokens/paso, bf16), RTX 5060 Laptop, PyTorch 2.11 + CUDA 12.8.

| Métrica | Manual | SDPA |
|---|---|---|
| Parámetros | 10,78M | 10,78M |
| Mejor val loss | 1,1289 | 1,1276 |
| Bits/char (fase 0 → fase 1) | 3,327 → 1,629 | 3,327 → 1,627 |
| Bits/byte (Quijote val, 1,028 bytes/char) | 1,584 | 1,582 |
| Paso del mejor checkpoint | 4500 | 4000 |
| Throughput medio (GPU fría, primeros 1000 pasos) | 81k tok/s (93k) | 143k tok/s (162k) |
| VRAM máxima (`max_memory_allocated`) | 2,61 GiB | 1,63 GiB |
| Duración aproximada | ~17 min | ~10 min |

Observaciones:

- **Contexto**: los bits/char bajan a la mitad respecto al bigrama (3,33 → 1,63). Es lo que aporta ver 256 caracteres en lugar de uno.
- **Overfitting**: desde el paso ~3000 la loss de train sigue bajando (0,96 → 0,83) y la de val se estanca en ~1,13. Los últimos 1000–1500 pasos no mejoran el modelo; por eso `best.pt` no es el último.
- **SDPA**: misma loss (las matemáticas son idénticas), ×1,76 de velocidad y un 38 % menos de VRAM, porque el kernel fusionado no materializa la matriz `T×T` de pesos de atención.
- **Térmica**: en el portátil la GPU llegó a 86 °C con *thermal slowdown* (reloj a 1.530 de 3.090 MHz). El throughput cae un ~12 % respecto a la GPU fría, y en entrenamientos de horas es el valor sostenido el que cuenta.

Muestra (SDPA, `best.pt`, temperatura 0,8, top-k 40):

```
En un lugar de la Mancha, dijo que en la calle y bebió las armas del amante
con los cuadrilleros andantes de los golpes y se le había vuelto en la
república; y así, sin duda, algunos por la reja que se llaman a la sepultura.
— ¡Bueno será Dios! —dijo don Quijote—. No sé yo de qué mal pensamiento tuviese,
```

## Estimación para Leon 0.1 con el throughput medido

El coste por token de entrenamiento es `6·N` (matmuls de los pesos, forward + backward) más `12·L·T·d` (la atención). Con la configuración de esta fase: 64,7M + 7,1M ≈ 72 MFLOP/token. A 143k tok/s salen **~10 TFLOP/s efectivos sostenidos** (~11,5 descontando la evaluación).

| Modelo × tokens | `6·N·D` | Horas a 10 TFLOP/s | Horas a 15 TFLOP/s |
|---|---|---|---|
| 30M × 1B | 1,8·10¹⁷ | ~5 h | ~3,3 h |
| 50M × 2B | 6,0·10¹⁷ | ~17 h | ~11 h |
| 60M × 3B | 1,1·10¹⁸ | ~30 h | ~20 h |

- La columna de 15 TFLOP/s es optimista: supone que un modelo más ancho (d = 512–640) aprovecha mejor la GPU que este de d = 384. La fase 3 lo medirá.
- Con contexto 1024 la atención deja de ser despreciable (+15–20 % sobre `6·N·D` en un modelo de 50M).
- **Conclusión provisional**: 50M × 2B (~2–3 noches) es viable; 60M × 3B empieza a ser caro. El tamaño final se fijará en la fase 3 con la arquitectura definitiva.
