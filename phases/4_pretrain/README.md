# Fase 4: preentrenamiento de Leon 0.1

## Objetivo

Entrenar el modelo base de Leon 0.1 con toda la Wikipedia en español: **56M de parámetros, contexto 1024 y ~2B de tokens**. Son unas 12,5 h de GPU (medidas), que se pueden repartir en varias sesiones y pausar y continuar en cualquier momento.

El resultado (`checkpoints/4_pretrain/leon_0.1/best.pt`) es un modelo que escribe español con fluidez pero aún no conversa. La conversación llega en la fase 5.

Después, una **extensión de contexto a 2048 tokens** (`control.py extend`, ~1,5 h): el mismo modelo, entrenado un poco más con ventanas el doble de largas. Ver [Extensión de contexto](#extensión-de-contexto-a-2048).

## Archivos

| Archivo | Contenido |
|---|---|
| `control.py` | `prepare`, `start`, `pause`, `resume`, `status`, `watch`, `extend`: maneja el entrenamiento como proceso en segundo plano |
| `train.py` | El bucle de entrenamiento, con la configuración de Leon 0.1 por defecto, pausa limpia y `status.json`. Con `--init-from`, `--rope-theta` y `--long-fraction`, también la extensión de contexto |
| [`leon/run.py`](../../leon/run.py) | Ficheros de control de un entrenamiento: PID, petición de pausa, estado, log y "no suspender Windows" |

El modelo, los datos y los checkpoints son los de las fases 2 y 3 (`leon.model`, `leon.data`, `leon.training`).

## Conceptos nuevos

- **Escala**: de 131M de tokens en la fase 3 a 2.000M. Con ~36 tokens por parámetro, Leon 0.1 se entrena más allá del óptimo de Chinchilla (~20). Cuesta más entrenarlo, pero el modelo resultante es mejor para su tamaño, y eso es lo que importa en un modelo que luego se usa mucho.
- **Batch de 131.072 tokens por paso**: 32 micro-batches de 4×1024 acumulados. 15.259 pasos dan 2.0B de tokens.
- **Schedule largo**: 500 pasos de warmup y decaimiento coseno de 1e-3 a 1e-4 hasta el paso 15.259. El schedule depende de `max_steps`, así que al reanudar se toma del checkpoint, nunca de la línea de comandos.
- **Entrenamiento por sesiones**:
  - **Proceso independiente**: el entrenamiento sigue aunque se cierre la terminal o el editor.
  - **Pausa limpia**: el fichero `PAUSE` hace que termine el paso en curso, guarde y salga.
  - **Reanudación exacta**: se restauran pesos, optimizador y generadores aleatorios, así que pausar y continuar da **exactamente** el mismo modelo que entrenar sin parar. Se comprobó con una ejecución de prueba comparando los pesos.
- **Seguimiento**: `status.json` (paso, pérdida, tok/s, ETA) cada 10 pasos, y una muestra de texto con el prompt fijo en cada evaluación (cada 250 pasos, ~13 min), para ver cómo aprende a escribir.
- **Sin suspensión**: mientras entrena, Windows no entra en suspensión por inactividad. Cerrar la tapa del portátil sí puede suspenderlo, según la configuración de energía.

## Ejecución

Desde la raíz del repo. Una vez, descarga y tokeniza toda la Wikipedia. Usa CPU y red, no GPU: ~3,5 GB de descarga, ~13 GB en disco y unos 20–40 min.

```powershell
python phases/4_pretrain/control.py prepare
```

Después, por sesiones:

```powershell
python phases/4_pretrain/control.py start     # la primera vez
python phases/4_pretrain/control.py status    # ¿cómo va? (una vez)
python phases/4_pretrain/control.py watch     # panel en vivo que se refresca solo
python phases/4_pretrain/control.py pause     # guarda y para (unos segundos)
python phases/4_pretrain/control.py resume    # continúa donde se quedó
```

El entrenamiento corre **sin ventana**, así que no hay nada que se pueda cerrar por error. `start` y `resume` abren además una ventana con el panel `watch` (título "Leon 0.1 · pretraining", en inglés como todo el código), que muestra:

- estado y barra de progreso;
- pérdidas y velocidad;
- la hora estimada de fin;
- la GPU;
- la última muestra de texto;
- las últimas líneas del log.

**Cerrar esa ventana no detiene el entrenamiento**: para pararlo, usa `pause`. Con `--no-watch` no se abre la ventana.

En una sesión con Claude Code basta con decir **"prepara los datos de la fase 4"**, **"arranca"**, **"¿cómo va?"**, **"pausa"** o **"continúa"** (ver [`CLAUDE.md`](../../CLAUDE.md)).

Otros usos:

- `start --fresh` borra el progreso y empieza de cero.
- `--run nombre` mantiene varias ejecuciones separadas.
- Lo que va después de `--` se pasa a `train.py`. Por ejemplo, una prueba rápida en CPU: `control.py start --run prueba -- --dataset quijote --device cpu --no-compile --n-layer 1 --n-embd 32 --n-head 2 --n-kv-head 1 --max-steps 300`.

## Extensión de contexto a 2048

El preentrenamiento usa 1024 tokens de contexto porque el coste de la atención crece con la longitud y la mayor parte de lo que se aprende (palabras, gramática, hechos) cabe en ventanas cortas. Duplicar el contexto desde el principio costaría bastante más GPU para casi la misma loss. Lo habitual (Llama 2 Long, Qwen, DeepSeek) es preentrenar corto y **extender al final** con una fracción pequeña de los tokens. Aquí, 151M de tokens, un 7,5 % del preentrenamiento.

```powershell
python phases/4_pretrain/control.py extend    # cuando `start` ha terminado
```

`pause`, `resume`, `status` y `watch` actúan sobre la última etapa iniciada, así que funcionan igual durante la extensión (`--run leon_0.1` vuelve a la base). El resultado queda en `checkpoints/4_pretrain/leon_0.1_ctx2048/best.pt`, y la fase 5 lo usa automáticamente si existe.

Qué hace `extend` (los parámetros están en `EXTEND_ARGS`, en `control.py`):

| Ajuste | Valor | Por qué |
|---|---|---|
| Pesos iniciales | `--init-from leon_0.1/best.pt` | Es el mismo modelo: solo cambian la longitud y RoPE, no hay parámetros nuevos |
| Contexto | `--block-size 2048` | Las tablas de RoPE se recalculan; los pesos no dependen de la longitud |
| Base de RoPE | `--rope-theta 50000` (antes 10000) | Con una base mayor, los canales lentos giran menos por posición: las posiciones 1024–2047 caen en ángulos parecidos a los que el modelo ya conoce, en vez de en ángulos que nunca vio (*ABF*, la técnica de Llama 2 Long y Code Llama) |
| Datos | `--long-fraction 0.5` | Media batch sale **de dentro de un único artículo de más de 2048 tokens** (137.753 en train; ~45 % de los tokens están en artículos así). Una ventana aleatoria suele juntar varios artículos cortos sin relación, y entonces el contexto lejano no ayuda a predecir nada |
| Schedule | 1.150 pasos de 2×2048×32 = 131.072 tokens, lr 3e-4 → 3e-5, warmup 100 | Por debajo del pico de la base (1e-3): ajustar lo aprendido, no reaprenderlo |

**La métrica clave**: en cada evaluación (cada 100 pasos), la loss en las posiciones 0–1023 y 1024–2047 de ventanas dentro de artículos largos (`leon.training.loss_by_position`; el panel la muestra como *Long text*).

- En un modelo que usa el contexto, la segunda mitad tiene **menos** loss: con más texto del mismo artículo detrás, predecir es más fácil.
- En el paso 0 (la base sin extender) la segunda mitad sale peor o igual, porque el modelo nunca vio esas posiciones.
- La extensión funciona si la segunda mitad termina claramente por debajo de la primera, y si la loss de val normal apenas empeora.

## Qué observar

1. **La loss al principio**: baja muy rápido en los primeros cientos de pasos (aprende frecuencias y palabras comunes) y después mucho más despacio. La mayor parte del entrenamiento se gasta en la "cola" lenta.
2. **Las muestras**: cada 250 pasos, del ruido inicial a palabras sueltas, frases gramaticales y párrafos con estilo de Wikipedia.
3. **Wikipedia frente al Quijote** en bits/byte: cuánto generaliza a un español distinto del que ve.
4. **tok/s en sesiones largas**: el *thermal throttling* del portátil (`status` muestra la temperatura).

## Ejercicios

- [ ] Comparar las muestras de los pasos 250, 1.000, 5.000 y 15.000. ¿En qué orden aprende ortografía, gramática y coherencia?
- [ ] Dibujar la curva de val loss desde `train.log` y ajustarle una ley de potencias `L(t) = a + b·t^(-c)`. ¿Cuánto mejoraría con el doble de tokens?
- [ ] Comparar `best.pt` con el modelo de 32M de la fase 3 en el chat (`apps/`), con los mismos prompts.
- [ ] Extensión: dibujar la loss por posición (de 64 en 64) del modelo base y del extendido sobre artículos largos. ¿Dónde se separan las curvas?
- [ ] Repetir la extensión sin `--rope-theta` (base 10000) o con `--long-fraction 0`. ¿Cuál de los dos ajustes importa más?

## Resultados

Preentrenamiento y extensión a 2048 terminados el 2026-11-07.

| Métrica | Valor |
|---|---|
| Tokens de entrenamiento (Wikipedia completa, tokenizer v2) | 1.357M train + 152M val (1,6M + 178k artículos, 3,90 bytes/token). 2,0B de tokens = 1,47 épocas |
| Mejor val loss / bits/byte (Wikipedia) | **2,4218 / 0,897** en el último paso (15.259). Fase 3, 32M: 1,175 bits/byte |
| Bits/byte en el Quijote (fuera de dominio) | **1,687** (paso 0: 4,055) |
| Throughput sostenido / horas totales de GPU | **~45.500 tok/s** (pasos 5.000 en adelante; ~43k al principio). **12 h 32 min** de reloj, incluidas compilación y evaluaciones. Pico de VRAM: 2,27 GiB |
| Sesiones | Una de 12,5 h sin pausas (2026-11-06 23:38 → 2026-11-07 12:10), tras reiniciar desde cero por el cambio a tokenizer v2 |
| Muestra con el prompt fijo | Ver abajo |
| Extensión a 2048: loss posiciones 0–1023 / 1024–2047, antes y después | Antes (paso 0, ya con RoPE 50000): 2,854 / 2,788 (−0,07). Después (paso 1.150): **2,732 / 2,560 (−0,17)** |
| Extensión a 2048: val loss y horas de GPU | Paso 0: 2,580 → final 2,432 (0,900 bits/byte); `best.pt` es el del paso 900, con 2,389 (0,884). Quijote: 1,696 → 1,657 bits/byte. **1 h 12 min** a ~37k tok/s (~33k al final, con la GPU a 87 °C) |

### Extensión a 2048

| Paso | Val loss | Bits/byte Quijote | Loss 0–1023 | Loss 1024–2047 | Diferencia |
|---|---|---|---|---|---|
| 0 | 2,580 | 1,696 | 2,854 | 2,788 | −0,07 |
| 100 | 2,458 | 1,676 | 2,813 | 2,700 | −0,11 |
| 300 | 2,506 | 1,674 | 2,817 | 2,684 | −0,13 |
| 500 | 2,476 | 1,670 | 2,772 | 2,622 | −0,15 |
| 900 | **2,389** | 1,659 | 2,707 | 2,537 | −0,17 |
| 1.150 | 2,432 | **1,657** | 2,732 | 2,560 | −0,17 |

- **Funciona**: la segunda mitad del contexto queda claramente por debajo de la primera, y la diferencia se duplica (−0,07 → −0,17). Con más texto del mismo artículo detrás, el modelo predice mejor: usa el contexto lejano.
- **Con RoPE 50000 no hace falta reaprender las posiciones**: ya en el paso 0, sin entrenar, la segunda mitad no se desmorona (sale algo mejor que la primera). Lo que se pierde al cambiar la base de RoPE (val 2,42 → 2,58) se recupera en 100 pasos.
- **La val loss es ruidosa** (±0,08 entre evaluaciones; en el paso 800 marcó 2,567). Sobre ventanas de 2048 tampoco se compara directamente con la de la base a 1024. La señal más estable es el Quijote, que baja en casi todas las evaluaciones.

### Curva de val loss

| Paso | Tokens | Val loss | Bits/byte Wikipedia | Bits/byte Quijote |
|---|---|---|---|---|
| 0 | 0 | 9,839 | 3,584 | 4,055 |
| 250 | 33M | 4,627 | 1,713 | 2,496 |
| 1.000 | 131M | 3,288 | 1,217 | 1,988 |
| 2.250 | 295M | 2,922 | 1,082 | 1,870 |
| 5.000 | 655M | 2,749 | 1,018 | 1,791 |
| 8.250 | 1,08B | 2,571 | 0,952 | 1,740 |
| 12.250 | 1,61B | 2,496 | 0,924 | 1,696 |
| 15.259 | 2,00B | **2,422** | **0,897** | **1,687** |

Cada evaluación usa pocos batches, así que la val loss oscila ±0,05 entre evaluaciones vecinas (en el paso 14.250 marcó 2,537). La tendencia es lo que cuenta.

### Muestras con "En un lugar de la Mancha"

- **Paso 250**: palabras y concordancias de Wikipedia, sin sentido. *«…las calles de la Iglesia católica, la estación del estado de la localidad estaba dedicado a la población. Según la carretera de 2010, el distrito de 1928 ganó la ciudad de San Carlos…»*
- **Paso 1.000**: frases gramaticales, pero se repite y no conecta los datos. *«…situado entre la población de Pilsen y la villa de Gruppe. A comienzos de la década de 1970, Pilsen pasó a ser uno de los mejores núcleos de población de Pilsen.»*
- **Paso 5.000**: párrafos con estructura de artículo. *«…se ha encontrado un fragmento de muralla de la Edad del Bronce que data del año 700 a.C. El actual recinto amurallado se encuentra en la cima del cerro del Poeta.»*
- **Paso 15.000**: estilo enciclopédico fluido, con nombres inventados pero plausibles. *«…un gran número de habitantes de la zona pueden encontrarse en el municipio de Fuente la Reina… existen otros asentamientos localizados en el municipio como La Torrecilla, el Puerto de la Zarza…»*

**Conclusión**: aprende primero la forma (ortografía, gramática, estilo y estructura) y después, mucho más despacio, la coherencia. Los datos los sigue inventando: es lo esperable con 56M de parámetros. Una muestra suelta es ruido; la del último paso salió peor que la del 15.000 con mejor loss.
