# Fundamentos: qué hay que saber para construir Leon 0.1

Guía de referencia para principiantes. Explica desde cero los conceptos necesarios para construir un modelo de lenguaje y define el objetivo del proyecto.

---

## 1. Qué es un LLM, en una frase

Un LLM es una función que recibe una secuencia de tokens y devuelve una distribución de probabilidad sobre el siguiente token. Todo lo demás (conversación, razonamiento, uso de tools) sale de aplicar esa función una y otra vez y de entrenarla con los datos adecuados.

```
"En un lugar de la"  →  [modelo]  →  P(siguiente token)
                                      "Mancha": 0,62
                                      "casa":   0,03
                                      ...
```

Generar texto es repetir tres pasos: muestrear un token de esa distribución, añadirlo a la secuencia y volver a llamar al modelo. Esto se llama **generación autorregresiva**.

---

## 2. El ciclo de vida completo

```
Texto bruto → Tokenizer → Preentrenamiento → Modelo base → SFT → Modelo chat → SFT tools → Modelo agéntico
                                                                                       ↑
                                                                         Runtime (bucle + tools)
```

| Etapa | Qué aprende el modelo | Datos |
|---|---|---|
| **Preentrenamiento** | El idioma, hechos del mundo, estructura del texto | Miles de millones de tokens de texto general |
| **SFT** (supervised fine-tuning) | El formato de conversación: turnos de usuario y asistente | Decenas de miles de diálogos |
| **SFT de tools** | Emitir llamadas estructuradas, leer resultados y responder | Diálogos con tool calls |
| **DPO / RLHF** | Preferir respuestas mejores frente a peores | Pares "buena / mala" |
| **Razonamiento** (SFT + RL con recompensa verificable) | Escribir los pasos intermedios antes de responder, y los que llevan a acertar | Problemas con solución comprobable (matemáticas, código) |

Lo esencial de esta tabla: **el modelo no "se vuelve" conversacional ni agéntico por arquitectura, sino por los datos con los que se ajusta.** La arquitectura es la misma en todas las etapas.

### El razonamiento

"Razonar" es que el modelo escriba pasos intermedios antes de la respuesta: sigue prediciendo el siguiente token, pero cada token cuesta un número fijo de cálculos (una pasada por las capas), y un problema como `17 × 24` no cabe en una sola pasada. Escrito como `17 × 20 = 340, 17 × 4 = 68, 340 + 68 = 408`, cada paso es fácil y queda en el contexto para el siguiente: más tokens es más cómputo.

Se consigue de tres formas, de menos a más esfuerzo:

1. **Pedirlo en el prompt** ("piensa paso a paso"). Funciona en modelos grandes, que vieron muchas explicaciones en el preentrenamiento.
2. **SFT con razonamientos**: problemas cuya respuesta incluye los pasos, a menudo escritos por un modelo mayor (destilación).
3. **Aprendizaje por refuerzo con recompensa verificable** (o1, DeepSeek-R1): el modelo genera varios intentos por problema, se premian los que aciertan, y se ajusta para que lo que llevó al acierto sea más probable. Nadie le dice *cómo* pensar: aprende solo a alargar el razonamiento, comprobar y corregirse. GRPO, el algoritmo de DeepSeek-R1, compara cada intento con la media de su grupo, sin un modelo de valor aparte.

El refuerzo no crea conocimiento: refuerza estrategias que el modelo ya produce alguna vez. Por eso brilla a partir de unos miles de millones de parámetros. Con los 56M de Leon 0.1, la fase 8 lo estudia en un terreno acotado, la aritmética, donde el efecto de escribir los pasos se puede medir.

---

## 3. Lo que hay que saber, por bloques

### A. Matemáticas (solo la parte que se usa)

- **Álgebra lineal:** vectores, matrices y producto matricial. Un transformer son casi solo multiplicaciones de matrices. Hay que razonar con shapes como `(batch, tiempo, dimensión)`.
- **Cálculo:** derivadas, regla de la cadena y gradiente. Entrenar consiste en calcular cómo cambia el error al mover cada parámetro y moverlo en la dirección que lo reduce. Esa operación es **backpropagation**, y la aplica PyTorch automáticamente (autograd). Aun así, hay que entender qué hace por dentro.
- **Probabilidad:** distribuciones y log-probabilidades.
  - **Softmax** convierte números arbitrarios (logits) en probabilidades.
  - **Cross-entropy** mide cuánta probabilidad asignó el modelo al token correcto. Es la loss que se minimiza en todas las fases.

No necesitas más que eso. La teoría avanzada (optimización convexa, teoría de la información a fondo) no hace falta para construirlo.

### B. Entrenamiento de redes neuronales

- **Parámetros:** los números que el modelo ajusta. "125M" significa 125 millones de ellos.
- **Forward / backward / step:** calcular la predicción, calcular los gradientes y actualizar los pesos con un optimizador (AdamW).
- **Learning rate y schedule:** el tamaño del paso. Es el hiperparámetro que más rompe un entrenamiento cuando está mal elegido.
- **Batch:** cuántos ejemplos se procesan por paso. Con **gradient accumulation** se simula un batch grande cuando falta VRAM.
- **Train vs. validación:** si la loss baja en train pero no en validación, hay **overfitting**: el modelo memoriza en lugar de generalizar.
- **Precisión numérica:** fp32 frente a bf16. bf16 reduce a la mitad la memoria y acelera, a cambio de un pequeño riesgo de inestabilidad.

### C. Tokenización

- Los modelos no ven caracteres ni palabras, sino **tokens**: fragmentos de texto frecuentes.
- **BPE** (Byte Pair Encoding) construye el vocabulario fusionando repetidamente el par de símbolos más frecuente.
- Importa porque un tokenizer entrenado en inglés parte el español en más tokens. El resultado es más coste y menos contexto útil.

### D. El transformer

Es el núcleo y el bloque que más tiempo merece. Piezas:

1. **Embeddings:** cada token se convierte en un vector.
2. **Self-attention:** cada token mira a los anteriores y decide cuáles son relevantes, mediante tres proyecciones llamadas *query*, *key* y *value*. Aquí está la clave de todo: permite usar contexto lejano, justo lo que le falta al bigrama de la fase 0.
3. **Máscara causal:** un token no puede ver los futuros. Sin ella, el modelo haría trampa durante el entrenamiento.
4. **Multi-head:** varias atenciones en paralelo, cada una especializada en un tipo de relación.
5. **MLP:** procesa cada posición por separado después de la atención.
6. **Residuales + normalización:** permiten apilar muchas capas sin que el entrenamiento se degrade.
7. **Posición:** la atención no sabe el orden por sí sola. Hay que inyectarlo, y la opción moderna es RoPE.

### E. Datos y escala

- **Scaling laws:** la calidad depende sobre todo de parámetros, tokens y cómputo. La referencia práctica (Chinchilla) es unos 20 tokens por parámetro como óptimo de cómputo.
- **Calidad de datos:** deduplicar, filtrar basura y detectar el idioma. Un corpus pequeño y limpio supera a uno grande y sucio.

### F. Post-entrenamiento y agéntica

- **Chat template:** tokens especiales que marcan quién habla.
- **Loss masking:** en SFT solo se entrena sobre las respuestas del asistente, no sobre lo que escribe el usuario.
- **Tool calling:** el modelo aprende a emitir algo como `<tool_call>{"name": "calculator", "args": {...}}</tool_call>`. El modelo nunca ejecuta nada: es el **runtime** quien parsea la llamada, ejecuta la tool y devuelve el resultado al contexto.
- Un **agente** es un bucle, no un modelo: el modelo decide, el runtime ejecuta, y se repite hasta obtener la respuesta final.

### G. Herramientas

- **Python + PyTorch:** tensores, `nn.Module`, autograd, mover datos a la GPU.
- **NumPy** para el procesamiento de datos.
- Saber leer curvas de loss y depurar shapes. Es el 50 % del trabajo real.

---

## 4. Qué bloque se aprende en cada fase

| Fase | Bloques |
|---|---|
| 0 Bigrama | A (softmax, cross-entropy), B (bucle de entrenamiento), G |
| 1 GPT mínimo | D completo, B (overfitting, learning rate) |
| 2 Tokenizer | C |
| 3 Arquitectura moderna | D (RoPE, normalización), B (bf16, eficiencia) |
| 4 Preentrenamiento | E, B a escala |
| 5 SFT | F (chat template, masking) |
| 6 Agente | F (tool calling, runtime) |

No hace falta estudiar todo antes de empezar. Cada fase introduce lo que necesita.

---

## 5. Objetivo del proyecto: Leon 0.1

### Objetivo de aprendizaje (el principal)

> Ser capaz de explicar e implementar desde cero cada componente del pipeline (tokenizer, transformer, preentrenamiento, SFT, tool calling y bucle agéntico) sin depender de librerías de alto nivel como Hugging Face Transformers para el modelo ni para el entrenamiento.

### Objetivo de producto (el entregable que lo demuestra)

> **Leon 0.1**: un modelo de unos 30–60M de parámetros, con tokenizer y preentrenamiento propios en español, entrenado íntegramente en hardware local (RTX 5060, 8 GB), que:
> 1. mantiene conversaciones cortas (3–5 turnos) gramaticalmente correctas y coherentes en temas generales sencillos, y
> 2. usa correctamente 2–3 tools simples (calculadora, fecha y hora, búsqueda en un conjunto de documentos locales) dentro de un bucle agéntico.

### Criterios de éxito medibles

Estos umbrales son orientativos. Se ajustarán cuando haya datos reales.

| Criterio | Meta |
|---|---|
| Tool calls con JSON válido según el schema | > 95 % |
| Elección correcta de tool en tareas del estilo del entrenamiento | > 80 % |
| Tareas agénticas sencillas completadas de principio a fin | > 60 % |
| Calidad conversacional | Evaluación manual con un set fijo de ~50 prompts, comparada fase a fase |

### Restricción de hardware

Todo se entrena en local, sin GPUs alquiladas. Para orientarse: el coste de entrenar es aproximadamente `6 × parámetros × tokens` FLOPs. Con 50M de parámetros y 2B de tokens salen ~6·10¹⁷ FLOPs. Medido en la fase 3 con la arquitectura definitiva y `torch.compile`, la RTX 5060 Laptop sostiene **~15 TFLOP/s efectivos** en bf16, incluso con *thermal throttling*. Para el tamaño elegido (56M de parámetros, contexto 1024) son ~42k tok/s, es decir, **~13–14 horas para 2B tokens**: dos noches. El detalle está en `phases/3_modern/README.md`.

### Expectativas honestas

- **Va a alucinar mucho en conocimiento factual.** Con este tamaño y estos datos, el modelo sabe poco del mundo. Su fuerte será el formato y la fluidez, no los hechos. Por eso la tool de búsqueda en documentos tiene sentido: compensa justo esa debilidad, y es la misma lógica que usan los sistemas reales (RAG).
- **El tool calling funcionará en tareas parecidas a las del entrenamiento**, no en tareas nuevas. Comprobar dónde está ese límite es en sí una de las lecciones del proyecto.
- **No se parecerá a ChatGPT.** La diferencia es de varios órdenes de magnitud en datos y cómputo, y entender por qué es otro de los aprendizajes.

---

## 6. Material de apoyo

- **Andrej Karpathy, *Neural Networks: Zero to Hero* (YouTube).** Cubre exactamente las fases 0–2 y es la mejor referencia práctica.
- **3Blue1Brown, serie sobre redes neuronales y transformers.** Para la intuición visual de los bloques A y D.
- **Paper *Attention Is All You Need* (2017).** Para leerlo después de la fase 1, no antes.
- **Documentación de PyTorch:** los tutoriales de tensores y autograd.

---

## Glosario rápido

| Término | Significado |
|---|---|
| Token | Unidad de texto que procesa el modelo (carácter, sílaba, palabra o fragmento) |
| Logits | Puntuaciones sin normalizar que el modelo asigna a cada token posible |
| Softmax | Función que convierte logits en probabilidades que suman 1 |
| Loss | Número que mide cuánto se equivoca el modelo; se minimiza al entrenar |
| Cross-entropy | La loss estándar en modelos de lenguaje: −log P(token correcto) |
| Gradiente | Dirección en la que cambiar los parámetros para reducir la loss |
| Backpropagation | Algoritmo que calcula los gradientes aplicando la regla de la cadena |
| Epoch / step | Pasada completa por los datos / una actualización de pesos |
| Overfitting | El modelo memoriza train y empeora en datos nuevos |
| Perplexity | `exp(loss)`: a cuántas opciones equiprobables equivale la incertidumbre del modelo |
| Bits por carácter / byte | Loss expresada en bits; permite comparar tokenizers distintos |
| Context window | Número máximo de tokens que el modelo ve a la vez |
| SFT | Ajuste supervisado con ejemplos de la conducta deseada |
| Tool call | Salida estructurada con la que el modelo pide al runtime ejecutar una función |
