# Guía de desarrollo

Cómo trabajar en el repositorio: preparar el entorno, el flujo de cada cambio y las convenciones. La organización del código está en [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Entorno

Windows, Python 3.11 o posterior, una GPU NVIDIA con CUDA 12.8+ para entrenar, y Node.js 20 o posterior para la web.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[all]"
cd apps/web; npm install; cd ../..
```

Los datos no están en el repositorio. Para reconstruirlos, sigue en orden la sección "Ejecución" de [`phases/0_bigram`](phases/0_bigram/README.md) y de [`phases/2_tokenizer`](phases/2_tokenizer/README.md).

## Antes de cada commit

```powershell
ruff check .            # lint
ruff format .           # formato
pytest                  # 34 tests, CPU, ~3 s; no necesitan data/ ni checkpoints/
python phases/1_gpt/test_model.py
cd apps/web; npm run build; cd ../..    # si cambió la web
```

Si cambia un notebook o el código que muestra, vuelve a ejecutarlo completo para que las salidas guardadas sigan siendo reales.

## Convenciones

- **Idiomas**: la documentación (README, ARCHITECTURE, notebooks) se escribe en español. El código, los comentarios, los identificadores y los mensajes de commit, en inglés.
- **Claridad antes que optimización**: el código es material de estudio. Solo se optimiza cuando una medición lo justifica, y la medición se documenta (ver `phases/3_modern/README.md`).
- **Abstracciones proporcionales**: nada de capas genéricas "por si acaso". El código pasa a `leon/` cuando lo usan dos sitios, no antes.
- **Cada afirmación con su medición**: los resultados de los README salen de ejecuciones reales, con el hardware y las versiones indicados.
- **Métricas comunes** en todas las fases, para poder comparar:
  - val loss y bits por byte (bits por carácter en las fases 0 y 1);
  - tok/s;
  - una muestra con el prompt fijo `En un lugar de la Mancha`;
  - el mismo split 90/10.
- **Entrenamientos reanudables**: todo script de entrenamiento guarda `last.pt` con `leon.training.save_checkpoint` y acepta `--resume`.

## Añadir una fase

1. Crea `phases/N_nombre/` con sus scripts, un `README.md` y un `walkthrough.ipynb`.
2. El README sigue la estructura de las fases anteriores:
   - objetivo;
   - archivos;
   - conceptos nuevos;
   - ejecución;
   - paso a paso;
   - qué observar;
   - ejercicios;
   - resultados.
3. Los checkpoints van a `checkpoints/N_nombre/<run>/`. El chat los detecta solo si la fase es la 3 o posterior.
4. El código que vayan a reutilizar otras fases o las apps se mueve a `leon/`, con tests en `tests/`.
5. Añade la fase a la hoja de ruta del [`README.md`](README.md) y una entrada en [`CHANGELOG.md`](CHANGELOG.md).

## Commits

- Mensajes en inglés y en imperativo: `Add phase 4 pretraining loop`, `Fix off-by-one in get_batch`.
- Un commit por cambio lógico, con la suite de calidad en verde.
- `data/`, `checkpoints/` y cualquier archivo de pesos están en `.gitignore`. Nunca deben entrar en el historial.
