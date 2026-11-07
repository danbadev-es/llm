# Leon 0.1 en GGUF

Exportado con `python -m leon.export checkpoints/7_dpo/leon_0.1_dpo/best.pt export/leon-0.1-<dtype>.gguf --dtype <dtype>`.

| Archivo | Tamaño | Precisión |
|---|---|---|
| `leon-0.1-q8_0.gguf` | 57 MB | 8 bits (recomendado) |
| `leon-0.1-f16.gguf` | 107 MB | 16 bits |
| `leon-0.1-f32.gguf` | 214 MB | completo |

Los `.gguf` no se versionan (`.gitignore`). La ficha del modelo, con el uso en LM Studio y Ollama, está en [`../MODEL_CARD.md`](../MODEL_CARD.md).
