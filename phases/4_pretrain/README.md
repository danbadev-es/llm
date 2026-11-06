# Fase 4: preentrenamiento de Leon 0.1

## Objetivo

Entrenar el modelo base de Leon 0.1 con toda la Wikipedia en español: **56M de parámetros, contexto 1024 y ~2B de tokens**. Son unas 13–14 h de GPU, repartidas en varias sesiones que se pueden pausar y continuar en cualquier momento.

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

*Pendiente de ejecutar.*

| Métrica | Valor |
|---|---|
| Tokens de entrenamiento (Wikipedia completa, tokenizer v2) | 1.357M train + 152M val (1,6M + 178k artículos, 3,90 bytes/token). 2,0B de tokens = 1,47 épocas |
| Mejor val loss / bits/byte (Wikipedia) | |
| Bits/byte en el Quijote (fuera de dominio) | |
| Throughput sostenido / horas totales de GPU | |
| Sesiones | |
| Muestra con el prompt fijo | |
| Extensión a 2048: loss posiciones 0–1023 / 1024–2047, antes y después | |
| Extensión a 2048: val loss y horas de GPU | |
