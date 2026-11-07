---
language:
  - es
pipeline_tag: text-generation
tags:
  - educational
  - from-scratch
  - llama-architecture
  - tool-calling
  - dpo
  - gguf
model_name: Leon 0.1
parameters: 55.96M
context_length: 2048
---

# Leon 0.1

Modelo de lenguaje conversacional en **español**, de **56 millones de parámetros**, construido y entrenado **desde cero** en un portátil (RTX 5060 Laptop, 8 GB) como proyecto educativo. Conversa, usa tres herramientas (calculadora, fecha y hora, búsqueda en documentos) y se ajustó por preferencias con DPO.

Es un modelo para **aprender cómo se hace un LLM**, no para usarlo como asistente: escribe español fluido, pero inventa los datos con seguridad y no sostiene un razonamiento abierto.

| | |
|---|---|
| Autor | Daniel Barreto |
| Fecha | noviembre de 2026 |
| Versión | 0.1, el modelo final: `checkpoints/7_dpo/leon_0.1_dpo/best.pt` |
| Código | este repositorio: tokenizer, modelo, entrenamiento y app escritos a mano, sin Hugging Face Transformers |
| Licencia | sin licencia publicada; ver [Datos y licencias](#datos-y-licencias) |

## Arquitectura

Un transformer decodificador al estilo Llama ([`leon/model.py`](leon/model.py)):

| | |
|---|---|
| Parámetros | **55,96M** (45,48M sin el embedding) |
| Capas | 10 |
| Dimensión | 640 |
| Atención | 10 cabezas de consulta, 5 de clave/valor (GQA), con KV cache |
| Posiciones | RoPE, base 50.000 (ampliada desde 10.000 para el contexto de 2048) |
| MLP | SwiGLU, 1.728 de dimensión oculta |
| Normalización | RMSNorm (pre-norm), ε = 1e-5 |
| Embedding | atado a la capa de salida |
| Contexto | 2.048 tokens |
| Tokenizer | BPE a nivel de byte, propio, 16.384 tokens (16 especiales), pre-tokenización de Llama 3 (v2): ~3,9 bytes por token en Wikipedia |

## Uso

**App local** (chat web y API compatible con la de Claude):

```powershell
python apps/api/app.py        # http://127.0.0.1:8000
```

Elegir `7_dpo/leon_0.1_dpo`, modo **Chat**, con el interruptor **Herramientas** activado. Temperatura recomendada: **0,5** (a 0,8 se desvía; por debajo de 0,4 se repite).

**LM Studio y Ollama**: el modelo exportado a GGUF (`python -m leon.export`):

| Archivo | Tamaño | Precisión |
|---|---|---|
| `leon-0.1-q8_0.gguf` | 57 MB | 8 bits: perplejidad +0,066 % frente a F32 (medido con llama.cpp) |
| `leon-0.1-f16.gguf` | 107 MB | 16 bits |
| `leon-0.1-f32.gguf` | 214 MB | completo |

El GGUF lleva el tokenizer y el formato de chat. En LM Studio basta con copiarlo a la carpeta de modelos. En Ollama:

```
FROM ./leon-0.1-q8_0.gguf
TEMPLATE """{{- range .Messages }}<|{{ .Role }}|>{{ .Content }}<|end|>{{ end }}<|assistant|>"""
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER temperature 0.5
```

Fuera de la app nadie ejecuta las herramientas: el modelo escribirá la llamada, pero no recibirá el resultado.

**Formato de chat**:

```
<|system|>Eres Leon, un asistente que responde en español de forma clara y breve.<|end|><|user|>¿Cuánto es 128 por 37?<|end|><|assistant|><|tool_call|>{"name": "calculadora", "arguments": {"expresion": "128*37"}}<|/tool_call|><|tool_result|>4736<|/tool_result|>El resultado es 4736.<|end|>
```

Las herramientas se anuncian en el *system prompt* (`leon.tools.system_prompt`); el *runtime* (`leon.agent`) ejecuta la llamada, añade el resultado y el modelo continúa el mismo turno.

## Entrenamiento

Cada etapa parte del modelo de la anterior. En total, **~16,5 h de GPU**.

| Etapa | Datos | Tiempo | Resultado |
|---|---|---|---|
| 1. Preentrenamiento ([fase 4](phases/4_pretrain/README.md)) | Wikipedia en español completa (dump de noviembre de 2023): **2.000M de tokens**, 1,47 épocas | 12,5 h | 0,897 bits/byte en Wikipedia, 1,687 en el Quijote |
| 2. Extensión de contexto 1024 → 2048 | 151M tokens, la mitad de dentro de artículos largos; RoPE 10.000 → 50.000 | 1,2 h | la segunda mitad del contexto, 0,17 nats por debajo de la primera |
| 3. SFT de conversación ([fase 5](phases/5_sft/README.md)) | 86.088 conversaciones: oasst2, aya, dolly, alpaca, WildChat (filas en español); pérdida solo en las respuestas | 1 h 53 min | pérdida de las respuestas 2,846 → 2,070 |
| 4. SFT de herramientas ([fase 6](phases/6_agent/README.md)) | 25.216 diálogos sintéticos con resultados de las herramientas reales, 30 % sin herramienta | 8 min | pérdida 2,157 → 0,874 |
| 5. DPO ([fase 7](phases/7_dpo/README.md)) | 4.504 pares: respuesta humana (buena) frente a una del propio Leon (mala), β = 0,1 | ~40 min | precisión de la recompensa 0,93 |

Optimizador AdamW, bf16, `torch.compile`, decaimiento coseno. Preentrenamiento: lr 1e-3, 131.072 tokens por paso. SFT: lr 1e-4. DPO: lr 1e-6 con referencia congelada.

## Evaluación

**Herramientas** ([`phases/6_agent/evaluate.py`](phases/6_agent/evaluate.py)), 450 preguntas que el modelo no vio, decodificación *greedy*, con el modelo de la etapa 4:

| | JSON válido | Herramienta correcta | Respuesta correcta |
|---|---|---|---|
| Calculadora | 100 % | 100 % | 99 % |
| Fecha y hora | 100 % | 100 % | 100 % |
| Documentos | 99 % | 99 % | 95 % |
| Búsqueda sin resultado | 100 % | 100 % | 44 % |
| Sin herramienta | 100 % | 99 % | 99 % |
| **Total** | **100 %** | **100 %** | **92 %** |

**DPO** ([`phases/7_dpo/evaluate.py`](phases/7_dpo/evaluate.py)), 62 preguntas nuevas, temperatura 0,8:

| | Antes del DPO | Después |
|---|---|---|
| Respuestas en bucle | 27 % | **16 %** |
| Respuestas que terminan solas | 79 % | **89 %** |
| Palabras por respuesta | 88 | 62 |

## Limitaciones

- **Inventa los datos** con total seguridad ("¿Qué colombianos han ganado el Nobel de paz?" → "Dolly Vásquez, reconocida poeta…"). Con 56M de parámetros sabe muy poco del mundo; la búsqueda en documentos existe para compensarlo.
- **Preguntas abiertas**: escribe frases gramaticales sin sentido ("¿Qué es la vida?" → "La vida es como una nube llena de agua…"). Respuestas coherentes a preguntas así aparecen hacia los 1.000M de parámetros.
- **Problemas con palabras**: usa la calculadora, pero solo traduce a una operación los tipos de problema que vio al entrenar. "Le quedan 96 chapas, que son 27 menos de las que tenía" → `96*3` (son 96 + 27).
- **Búsquedas irrelevantes**: si la búsqueda devuelve un pasaje que no responde a la pregunta, lo cita en vez de decir que no lo encuentra (44 % de aciertos en ese caso).
- **Historial**: tiende a copiar números y frases de los mensajes anteriores; las pruebas sueltas salen mejor en una conversación nueva.
- **Solo español**, y el español de Wikipedia y de los conjuntos de instrucciones: poco registro coloquial y regional.
- **Aritmética sin calculadora**: falla casi cualquier multiplicación de dos cifras. La [fase 8](phases/8_reasoning/README.md) le enseñó a resolverlas paso a paso, pero ese modelo (especializado, conversa peor) no es el final.

## Usos previstos y no previstos

- **Previstos**: estudiar cómo funciona y cómo se entrena un LLM, de punta a punta; experimentar con el código de cada fase; probar en local *tool calling*, DPO o cuantización con un modelo que cabe en cualquier sitio.
- **No previstos**: cualquier uso en el que importe que la respuesta sea cierta (información, consejos médicos, legales o financieros), o un producto.

## Datos y licencias

| Datos | Licencia |
|---|---|
| Wikipedia en español (`wikimedia/wikipedia`, 20231101.es) | CC BY-SA 4.0 |
| OpenAssistant oasst2 | Apache 2.0 |
| CohereLabs aya_dataset | Apache 2.0 |
| databricks-dolly-15k (traducción curada) | CC BY-SA 3.0 |
| bertin-project/alpaca-spanish | CC BY 4.0, generado con modelos de OpenAI |
| allenai/WildChat-1M | ODC-BY, conversaciones con ChatGPT |

Alpaca y WildChat proceden de modelos de OpenAI: aptos para estudiar, no para un producto comercial. Los diálogos de herramientas, los pares de DPO y los problemas de aritmética se generaron en el propio proyecto.

## Reproducir

Todo el proceso está en el repositorio, fase a fase, con los comandos y los resultados medidos: [README.md](README.md) (estado y guía), [ARCHITECTURE.md](ARCHITECTURE.md) (diseño) y los `README.md` de cada `phases/N_*/`. Los datos se regeneran con los `prepare.py` de cada fase, y cada entrenamiento largo se lanza, pausa y sigue con su `control.py`.
