"""Leon: the reusable code of the project, graduated from the phases where it was built.

    leon.tokenizer   byte-level BPE tokenizer           (built in phase 2)
    leon.model       Leon transformer + KV cache        (built in phase 3)
    leon.data        token files read with np.memmap    (built in phase 3)
    leon.chat        chat template                      (trained in phase 5, used by apps/api)
    leon.paths       where data and checkpoints live

Install once from the repo root with `pip install -e .`; then `from leon.model import Leon`
works from any script, notebook or app without touching sys.path.
"""

__version__ = "0.1.0"
