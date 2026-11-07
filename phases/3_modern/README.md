# Fase 3: arquitectura moderna

## Objetivo

Pasar del GPT de 2019 de la fase 1 a la arquitectura que usan los modelos actuales (Llama, Mistral, Qwen), y **medir el throughput real** de esa arquitectura en la RTX 5060 para fijar el tamaño de Leon 0.1.

Por primera vez el modelo se entrena con el tokenizer BPE de la fase 2, leyendo los `.bin` de Wikipedia con `np.memmap`.

## Archivos

| Archivo | Contenido |
|---|---|
| [`leon/model.py`](../../leon/model.py) | `Leon`: `RMSNorm`, RoPE (`precompute_rope`, `apply_rope`), `Attention` con GQA, `SwiGLU`, `KVCache`, `stream`/`generate`. Vive en el paquete `leon` porque es la arquitectura de Leon 0.1 que usan las fases 4–8 y el chat |
| [`leon/training.py`](../../leon/training.py) | `lr_at`, `bits_per_byte`, `save_checkpoint`/`load_checkpoint`: guardado atómico con estado del RNG, para que `--resume` continúe exactamente donde se quedó |
| [`leon/data.py`](../../leon/data.py) | `TokenFile`: lee los `.bin` uint16 con `np.memmap` y sabe sus bytes/token (para bits por byte) |
| `train.py` | Bucle de entrenamiento de la fase 1 con datos en disco, bits/byte, TFLOP/s y evaluación cruzada en el Quijote |
| `sample.py` | Generación desde un checkpoint, con y sin KV cache |
| `benchmark.py` | Mide tok/s, TFLOP/s y VRAM de varios tamaños candidatos de Leon 0.1 |
| [`tests/test_model.py`](../../tests/test_model.py) | Comprobaciones: RMSNorm, propiedades de RoPE, causalidad, GQA frente a atención manual, KV cache frente a forward completo |

## Conceptos nuevos respecto a la fase 1

- **RMSNorm**: `x / rms(x) · g`. Como LayerNorm pero sin restar la media ni sumar un bias. Más barato y funciona igual de bien.
- **RoPE (Rotary Position Embedding)**: en lugar de sumar un vector de posición, se **rota** cada par de canales de `q` y `k` un ángulo proporcional a la posición. Como las rotaciones se componen, `q_m · k_n` solo depende de `m - n`: el modelo ve **distancias relativas**. `test_rope_preserves_norm_and_is_relative` lo comprueba.
- **SwiGLU**: `down(silu(gate(x)) · up(x))`. Una compuerta aprendida decide, por token y por unidad, cuánto pasa. Tiene tres matrices en lugar de dos, así que la capa oculta se reduce a ~8/3·d para mantener el coste.
- **GQA (Grouped-Query Attention)**: 8 cabezas de query comparten 4 cabezas de key/value. Durante el entrenamiento ahorra poco; en generación reduce a la mitad el **KV cache**.
- **KV cache**: al generar, las keys y values de los tokens anteriores no cambian. Se guardan y cada paso solo procesa el token nuevo. `test_kv_cache_matches_full_forward` comprueba que da exactamente los mismos logits.
- **Sin biases** en las capas lineales.
- **`ignore_index=-100`** en la loss: aquí no se usa; en la fase 5 (`leon/sft.py`) es lo que evita entrenar sobre los turnos del usuario.
- **Bits por byte**: `loss / ln 2 / (bytes/token)`. Con un tokenizer distinto, es la única forma de comparar con las fases 0 y 1.
- **El coste del vocabulario**: con 16.384 tokens, los logits de un micro-batch de 16k tokens son 268M de números. En fp32, con su log-softmax y su gradiente, ocupan ~3,5 GiB. Por eso los micro-batches son pequeños (4.096 tokens) y el batch efectivo se consigue con gradient accumulation.
- **`torch.compile`**: fusiona las operaciones elemento a elemento (RMSNorm, RoPE, SwiGLU, softmax de la loss) en menos kernels, con menos lecturas y escrituras de memoria. En Windows necesita `triton-windows` (incluido en `pip install -e ".[compile]"`).

## Ejecución

Desde la raíz del repo (requiere haber completado la fase 2):

```powershell
pytest tests/test_model.py tests/test_training.py tests/test_data.py
python phases/3_modern/benchmark.py --compile    # tamaños candidatos de Leon 0.1
python phases/3_modern/train.py --compile        # 32M en Wikipedia, 2000 pasos de 65k tokens
python phases/3_modern/sample.py --prompt "La ciudad de Madrid"
python phases/3_modern/sample.py --no-cache      # comparar velocidad sin KV cache
```

### Paso a paso

`walkthrough.ipynb` aísla cada pieza nueva con números pequeños:

- RMSNorm frente a LayerNorm.
- RoPE como rotación, y por qué `q·k` solo depende de la distancia.
- SwiGLU frente a GELU.
- GQA y el tamaño del KV cache.
- El KV cache en acción: misma salida, menos tiempo.
- La memoria de los logits con un vocabulario de 16k.
- Batches con `np.memmap`.

Corre en CPU en segundos.

## Qué observar

1. **Memoria del vocabulario**: `benchmark.py --tokens-per-micro-batch 16384` frente a `4096`. Si la VRAM pasa de 8 GiB, Windows desborda a la RAM compartida y la velocidad cae un orden de magnitud sin dar error.
2. **TFLOP/s casi constantes**: entre 20M y 63M la GPU rinde lo mismo por FLOP. Lo que cambia es cuántos FLOPs cuesta cada token.
3. **Bits por byte en Wikipedia frente al Quijote**: el modelo se entrena con Wikipedia, y el Quijote es otro dominio.
4. **KV cache**: los tok/s de generación con y sin cache.

## Ejercicios

- [ ] Sustituir RoPE por embeddings de posición aprendidos (como en la fase 1) y comparar la val loss a igual número de pasos.
- [ ] Probar `--n-kv-head 8` (atención multi-head completa) y `--n-kv-head 1` (multi-query). ¿Cambia la loss? ¿Y el tamaño del KV cache?
- [ ] Generar más allá de `block_size` sin cache. ¿Qué pasa con RoPE en posiciones que nunca vio?
- [ ] Medir con `benchmark.py` el efecto de `--compile` en cada tamaño.

## Resultados

RTX 5060 Laptop, PyTorch 2.11 + CUDA 12.8, bf16, `torch.compile`.

### Velocidad final de Leon 0.1 (d = 640, 10 capas, GQA 10/5, contexto 1024)

Medida con la GPU fresca, tras la revisión de código.

| | Valor |
|---|---|
| Entrenamiento (`torch.compile`, bf16, micro-batch de 4.096 tokens) | **41.946 tok/s, 17,4 TFLOP/s**, 2,05 GiB de VRAM |
| Tiempo para 2B tokens | **~13 h** en frío, ~14 h contando el *thermal throttling* |
| Prefill (procesar el prompt) | 12.140 tok/s (128 tokens), 52.639 (512), **87.583 (1.000)** |
| Generación, 1 conversación | **94 tok/s** (10,7 ms por token) |
| Generación por lotes | 733 tok/s (8 a la vez), **3.001 tok/s** (32 a la vez) |
| Pesos en bf16 / VRAM total en inferencia | 107 MiB / 748 MiB |

La generación de una sola conversación está limitada por la sobrecarga de lanzar kernels, no por el cálculo (ver "KV cache: CPU frente a GPU"). Por eso al generar 32 conversaciones a la vez el total sube ×32 casi gratis.

### Benchmark de tamaños candidatos (`benchmark.py --compile`, 4.096 tokens por micro-batch)

| Candidato | Parámetros | tok/s (ctx 512) | tok/s (ctx 1024) | TFLOP/s | VRAM | Horas para 2B tokens (ctx 1024) |
|---|---|---|---|---|---|---|
| d384 · L8 | 18,9M | 118.511 | 96.710 | 15,7 | 1,1 GiB | 5,7 |
| d512 · L8 | 32,0M | 71.392 | 64.187 | 15,5 | 1,4 GiB | 8,7 |
| d512 · L12 | 43,8M | 51.734 | 46.016 | 15,5 | 1,9 GiB | 12,1 |
| d640 · L10 | 56,0M | 45.732 | 41.946 | 17,4 | 2,1 GiB | 13,2 |
| d768 · L8 | 62,9M | 39.871 | ~37.000* | 16,6 | 2,1 GiB | ~15,0 |

\* La medición en ctx 1024 de este candidato coincidió con *thermal throttling*; el valor se estima a partir de ctx 512 con la misma caída que los demás (~5–10 %). La fila d640 · L10 es la medición final, repetida con la GPU fresca.

- **Sin compilar**, los mismos modelos rinden ~10 TFLOP/s: `torch.compile` aporta un ×1,5–1,6.
- **`enable_gqa=True` en SDPA** (evita copiar K y V para cada grupo de cabezas) resultó un ~20 % **más lento** y con más memoria en esta configuración: sale del kernel fusionado de flash attention. Se mantiene `repeat_interleave`.
- **Con micro-batches de 16.384 tokens** los logits no caben: la VRAM pasa de 8 GiB, Windows desborda a la RAM compartida y la velocidad cae de 10 a 30 veces.

### Validación: 32M en Wikipedia (`train.py --compile`, configuración por defecto)

| Métrica | Valor |
|---|---|
| Parámetros | 31,99M (23,60M sin embedding) |
| Tokens vistos | 131M (2.000 pasos × 65.536) en 31 min |
| Mejor val loss (Wikipedia) | 3,3479 nats/token = **1,175 bits/byte** (paso 2000, sigue bajando) |
| Quijote val (fuera de dominio) | 1,942 bits/byte |
| Throughput sostenido | **70.620 tok/s, 15,3 TFLOP/s** (media de 20 intervalos, GPU a 85–87 °C) |
| VRAM máxima | 1,55 GiB |

Muestra (`best.pt`, temperatura 0,8, top-k 40):

```
En un lugar de la Mancha, el rey Enrique II de Aragón, y en la provincia de Cuenca y en la que ha tenido
la conquista de las costas del reino, se ha conservado en una gran muralla.

Al igual que en la parte baja (1547), la muralla de la Mancha ha sido de interés territorial, por lo que ha
pasado a ser muy apreciada en la zona. La muralla de la Mancha es la más antigua de Castilla...
```

Gramática y estilo de enciclopedia correctos, hechos inventados. Es lo esperable en un modelo base pequeño.

### KV cache: CPU frente a GPU

Generación con el modelo de 32M (tok/s, temperatura 0,8, top-k 40):

| | 100 tokens | 400 tokens |
|---|---|---|
| CPU sin cache | 62 | 22 |
| CPU con cache | 135 | 131 |
| GPU sin cache | — | 122 |
| GPU con cache | — | 116 |

- **En CPU** el cache es un ×6 a 400 tokens, y la ventaja crece con la longitud: sin cache, cada token nuevo recalcula todo el contexto (coste cuadrático).
- **En GPU no se nota**: con un modelo tan pequeño, cada token lanza ~200 kernels diminutos y el tiempo se va en la sobrecarga de Python y de los lanzamientos (~8 ms/token), no en el cálculo. El cache ahorra un cálculo que no era el cuello de botella.
- Lo que acelera la decodificación en GPU es reducir esa sobrecarga (CUDA graphs, `torch.compile(mode="reduce-overhead")`) o generar varias secuencias a la vez. Con ~120 tok/s, de todos modos, el chat va 20 veces más rápido de lo que se lee.

### Comparación con la fase 1 en el Quijote

Misma escala que la fase 1 (6 capas, d = 384, contexto 256, dropout 0,2), entrenado con el Quijote tokenizado con BPE:

| | Fase 1 (caracteres, GPT-2) | Fase 3 (BPE, arquitectura moderna) |
|---|---|---|
| Parámetros | 10,8M | 15,7M (6,3M son el embedding de 16k tokens) |
| Tokens de entrenamiento | 1,9M | 0,56M |
| Mejor val (Quijote) | **1,582 bits/byte** (paso 4500) | 1,612 bits/byte (paso 500) |
| Overfitting al final | train 0,83 / val 1,13 nats | train 1,46 / val 4,28 nats |

**Con un corpus de 2 MB, BPE + arquitectura moderna no gana.** Con 16k tokens distintos y solo 558k de entrenamiento, cada token se ve pocas veces y el modelo memoriza a partir del paso 500 (~15 épocas). Las mejoras de esta fase pagan con **muchos datos**: en Wikipedia, el mismo código baja sin señales de overfitting. Es la razón de que Leon 0.1 se entrene con Wikipedia.

### Tamaño de Leon 0.1 (decidido)

**d = 640, 10 capas, 10 cabezas de query y 5 de key/value, contexto 1024: 56M parámetros (45,5M sin embedding), con ~2B tokens.**

- **Coste**: ~13–14 h de GPU a ~42k tok/s (medición final, ver arriba), es decir, **2 noches** con `--resume`. En la validación de 32M, el throughput sostenido estuvo a un 1–5 % del benchmark en frío.
- **Tokens por parámetro**: ~36, por encima del óptimo de Chinchilla (20). Para un modelo que se va a usar mucho en inferencia conviene entrenar más allá de ese punto.
- **Contexto 1024**: las fases 5 y 6 necesitan conversaciones de 3–5 turnos más resultados de herramientas. Cuesta un ~10 % más que 512.
- **Datos**: Wikipedia completa da ~1,4B tokens, así que 2B tokens son ~1,4 épocas. Repetir datos hasta ~4 épocas apenas perjudica, así que la fase 4 se quedó solo con Wikipedia (1,47 épocas).
- **Alternativa más barata**: d = 512 con 12 capas (44M), ~12 h. Las profundidades mayores suelen rendir mejor a igual tamaño en modelos pequeños, así que es un candidato serio si se quiere ahorrar una noche.

