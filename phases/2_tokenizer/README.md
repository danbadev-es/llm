# Fase 2: tokenizer BPE propio

## Objetivo

Sustituir el tokenizer de caracteres por un **BPE a nivel de byte** entrenado con Wikipedia en español. Es el tokenizer definitivo de Leon 0.1: se usa sin cambios desde el preentrenamiento (fase 4) hasta el razonamiento (fase 8), en su versión v2 (ver más abajo).

Además, esta fase deja los corpus **ya tokenizados en disco** (`.bin` de uint16), listos para leerse con `np.memmap` sin cargarlos en RAM.

## Archivos

| Archivo | Contenido |
|---|---|
| [`leon/tokenizer.py`](../../leon/tokenizer.py) | `BPETokenizer`: pre-tokenización por regex, entrenamiento incremental, encode/decode, tokens especiales, save/load. Vive en el paquete `leon` porque lo usan todas las fases posteriores |
| `prepare.py` | Descarga shards parquet de Wikipedia (dump 20231101.es) y los convierte en `train.jsonl` / `val.jsonl` |
| `train_tokenizer.py` | Entrena el tokenizer con una muestra de Wikipedia |
| `encode.py` | Tokeniza Wikipedia y el Quijote a `.bin` (uint16) en paralelo y en streaming |
| `inspect_tokenizer.py` | Muestra cómo se parte el texto, los merges aprendidos y la compresión |
| [`tests/test_tokenizer.py`](../../tests/test_tokenizer.py) | Comprobaciones: split sin pérdidas, equivalencia con un BPE ingenuo, roundtrip, tokens especiales, save/load |

## Conceptos nuevos

- **BPE (Byte Pair Encoding)**: se parte de los 256 bytes y se fusiona una y otra vez el par adyacente más frecuente. Cada fusión (*merge*) crea un token nuevo. Con 16.384 tokens, las palabras frecuentes (` de`, ` población`) acaban siendo un único token y las raras se componen de trozos.
- **Nivel de byte**: como el vocabulario contiene los 256 bytes, **cualquier texto** se puede codificar (emoji, chino, errores tipográficos). No existe el token "desconocido".
- **Pre-tokenización**: antes de BPE, una regex parte el texto en trozos (`" lugar"`, `","`, `" 23"`). Los merges nunca cruzan esos límites, así que no aparecen tokens absurdos como `"a, d"`. El espacio va pegado al principio de la palabra.
- **Números de 1–3 dígitos**: `1234` → `[ 123][4]`. Evita que el vocabulario se llene de números concretos y da al modelo una representación más regular de la aritmética.
- **Normalización NFC**: `á` puede escribirse como 1 o 2 code points. Normalizar evita que la misma palabra tenga dos secuencias de bytes.
- **Entrenamiento incremental**: recontar todos los pares tras cada merge costaría O(corpus) por merge. `leon/tokenizer.py` actualiza solo las palabras que contienen el par fusionado y usa un heap con *lazy deletion*. `tests/test_tokenizer.py` comprueba que da **exactamente** los mismos merges que la versión ingenua.
- **Entrenar sobre frecuencias, no sobre texto**: BPE solo necesita la tabla `trozo → frecuencia`. 100M de caracteres se reducen a unos cientos de miles de trozos distintos.
- **Tokens especiales reservados**: `<|endoftext|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|tool_call|>`, `<|tool_result|>`... Se reservan ya para que el vocabulario no cambie en las fases 5 y 6. Por defecto `encode` **no** los reconoce dentro del texto: un artículo que contenga la cadena `<|user|>` no puede inyectar un token de control.
- **Bits por byte**: la loss por token no se puede comparar entre tokenizers, porque un token BPE contiene más texto que un carácter. La medida comparable es `bits/byte = loss · tokens / (bytes · ln 2)`. `encode.py` guarda `tokens` y `bytes` de cada fichero en su `.json`.

## Ejecución

Desde la raíz del repo:

```powershell
python tests/test_tokenizer.py
python phases/2_tokenizer/prepare.py                 # shard 0: ~690 MB de descarga, ~1,1 GB de texto
python phases/2_tokenizer/train_tokenizer.py         # vocab 16.384, 100M caracteres
python phases/2_tokenizer/inspect_tokenizer.py
python phases/2_tokenizer/encode.py
```

Para más datos: `prepare.py --shards 0 1 2`. Para comparar tamaños de vocabulario: `train_tokenizer.py --vocab-size 8192 --output data/tokenizer/bpe8k.json` e `inspect_tokenizer.py --tokenizer data/tokenizer/bpe8k.json`.

El tokenizer queda en `data/tokenizer/leon_bpe_v2.json` (la versión predeterminada, `TOKENIZER_PATH` en `leon/paths.py`), con `leon_bpe_v2.vocab.txt` al lado para leer el vocabulario token a token.

### Paso a paso

`walkthrough.ipynb` ejecuta BPE a mano sobre un corpus de juguete, muestra un encode merge a merge, la curva de compresión frente al número de merges, la conversión a bits por byte y la lectura con `np.memmap`. Corre en CPU en menos de un minuto; en VS Code, selecciona el kernel de `.venv`.

## Qué observar

1. **Orden de los merges**: los primeros son fragmentos ultrafrecuentes del español (`de`, ` e`, ` de`, ` l`, `os`). Los últimos son palabras o raíces de frecuencia media (` sintaxis`, ` monje`, ` neocl`). Entre los tokens más largos hay uno de 32 espacios: Wikipedia tiene sangrías y tablas, y el tokenizer refleja fielmente los datos, incluidas sus rarezas.
2. **Compresión**: bytes/token en Wikipedia frente al Quijote. El tokenizer comprime mejor el dominio con el que se entrenó; el castellano del siglo XVII (`vuestra merced`, `dél`, `hidalgo`) se parte más.
3. **Palabras tras `¿` y `¡`**: `¿Cuánto` se parte peor que ` cuánto`, porque la palabra queda sin el espacio inicial con el que suele aparecer. Es una peculiaridad del español que la regex de GPT (pensada para inglés) no contempla.
4. **Contexto efectivo**: con ~4 bytes/token, 256 tokens de contexto cubren ~1.000 caracteres, cuatro veces más que en la fase 1 con el mismo coste de atención.
5. **Coste del embedding**: con `n_embd = 512`, un vocabulario de 16.384 son 8,4M de parámetros solo en la tabla de embedding (compartida con la salida). En un modelo de 40M es una fracción grande, y es la razón de no subir a 32k o 50k.

## Ejercicios

- [ ] Entrenar con `--vocab-size 4096`, `8192` y `32768`, y apuntar bytes/token en Wikipedia val y en el Quijote. ¿Dónde se aplana la curva?
- [ ] Entrenar con `--train-mchars 10` frente a `100`. ¿Cambia mucho la compresión? ¿Por qué basta con una muestra?
- [ ] Quitar la pre-tokenización (usar un patrón que devuelva el texto entero) y ver qué tokens aparecen en `leon_bpe_v2.vocab.txt`.
- [ ] Cambiar `\d{1,3}` por `\d+` y mirar cuántos tokens del vocabulario acaban siendo números.
- [ ] Modificar la regex para que `¿` y `¡` no dejen la palabra siguiente sin espacio, y medir el efecto en la compresión.
- [ ] Calcular a mano los bits/byte del modelo de la fase 1 a partir de sus bits/char y de los bytes/char del Quijote.

## Resultados

| Métrica | Valor |
|---|---|
| Vocabulario | 16.384 (256 bytes + 16.112 merges + 16 especiales) |
| Texto de entrenamiento / tiempo | 100M caracteres (448k trozos únicos) / 30 s (6 s pre-tokenizar + 24 s merges) |
| Bytes/token Wikipedia val | 4,19 (4,11 contando un `<\|endoftext\|>` por artículo) |
| Bytes/token Quijote val | 3,50 |
| Tokens/palabra Wikipedia val / Quijote val | 1,54 / 1,62 |
| Velocidad de encode (1 proceso / pool de 14) | 3,8 MB/s / 23,5 MB/s |
| Tokens en disco (shard 0) | 253,8M train + 28,4M val (541 MB en uint16) |
| Fase 1 en bits/byte (Quijote val) | 1,582 (SDPA) |

Observaciones:

- El tokenizer comprime un 20 % peor el Quijote que Wikipedia: es otro dominio (castellano del siglo XVII).
- Con 4,1 bytes/token, el shard 0 (688 MB de parquet) da 282M tokens. Los 13 shards (3,49 GB) darían **~1,4B tokens**, dentro del rango de 1–3B previsto para Leon 0.1.
- Para igualar los 1,582 bits/byte de la fase 1 en el Quijote, un modelo con este tokenizer necesita una val loss de 1,582 · 3,50 · ln 2 ≈ **3,84 nats/token**. Esa es la referencia para la fase 3.

### Tokenizer v2: compatible con llama.cpp (LM Studio, Ollama)

Las tablas de arriba son del **v1**, el primer tokenizer, que usan los modelos de la fase 3. Para poder exportar Leon a GGUF y usarlo en LM Studio u Ollama, el tokenizer tiene que partir el texto **exactamente** igual que llama.cpp, y llama.cpp solo admite unas reglas de pre-tokenización concretas. Medido con el v1:

| Regla de llama.cpp aplicada a los merges del v1 | Documentos tokenizados igual | Diferencia de tokens |
|---|---|---|
| GPT-2 | 0 % | +2,5 % |
| Llama 3 (`llama-bpe`) | 10,8 % | +1,9 % |

Por eso el **v2** se entrena con los mismos datos y el mismo tamaño de vocabulario, pero con la regla de pre-tokenización de **Llama 3**, idéntica a la de llama.cpp (`PRETOKENIZE_PATTERN` en `leon/tokenizer.py`). Usa el módulo `regex`, porque el `re` de Python no entiende `\p{L}` ni `\p{N}`.

| | v1 | v2 |
|---|---|---|
| Bytes/token Wikipedia val | 4,19 | 4,11 (−1,8 %) |
| Bytes/token Quijote val | 3,50 | 3,50 |
| `1234` | ` 123` · `4` | ` ` · `123` · `4` (el espacio va aparte) |
| Compatible con llama.cpp | ~98 % de los tokens | 100 % |
| Tokens en disco, Wikipedia completa | — | 1.357M train + 152M val |

- **Versiones**: `leon/paths.py` registra las dos (`TOKENIZERS`), y el v2 es el predeterminado.
- **Cada checkpoint guarda su versión** (clave `tokenizer`), así que el chat usa la correcta: los modelos de la fase 3 son v1, y Leon 0.1 es v2 desde la fase 4.
- **Una lección**: el tokenizer se elige pensando también en dónde se va a ejecutar el modelo. Cambiarlo obliga a reentrenar todo lo que viene después, y por eso se hizo con la fase 4 al 10 %.

