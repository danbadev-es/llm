"""Leon's tools (phase 6): what the model can ask the runtime to do.

    calculadora(expresion)       exact arithmetic, the thing a 56M model is worst at
    fecha_hora()                 today's date and the time, which no model can know
    buscar_documentos(consulta)  passages of the user's documents in data/docs/ (BM25)

The model never runs anything: it writes a call as JSON, and leon.agent parses it, runs the tool
here and puts the result back in the context. A tool always returns text and never raises:
a bad call becomes an "error: …" result the model can read and recover from.
"""

import ast
import heapq
import math
import operator
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from leon.chat import DEFAULT_SYSTEM
from leon.paths import DATA_DIR

DOCS_DIR = DATA_DIR / "docs"
MAX_RESULT_CHARS = 700  # a search result must leave room for the answer in a 2048-token context


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]  # argument name -> description (all strings, all required)
    run: Callable[[dict], str] | None  # None: a client tool, run by whoever called the API

    @property
    def signature(self) -> str:
        return f"{self.name}({', '.join(self.parameters)})"

    def __call__(self, arguments: dict) -> str:
        if not isinstance(arguments, dict):
            return "error: los argumentos deben ser un objeto JSON"
        missing = [name for name in self.parameters if name not in arguments]
        if missing:
            return f"error: falta el argumento {missing[0]}"
        if self.run is None:
            return "error: esta herramienta la ejecuta el cliente"
        try:
            return self.run(arguments)
        except Exception as error:  # a tool must never break the conversation
            return f"error: {error}"


# ---------------------------------------------------------------- calculadora

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"raiz": math.sqrt, "sqrt": math.sqrt, "abs": abs, "redondear": round, "round": round}
_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPRESSION = 120
MAX_MAGNITUDE = 1e15


def _normalize_expression(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("×", "*").replace("·", "*").replace("÷", "/").replace("^", "**").replace("−", "-")
    return re.sub(r"(?<=\d),(?=\d)", ".", text)  # 3,5 (Spanish decimals) -> 3.5


def _evaluate(node: ast.AST) -> float:
    """Evaluate an arithmetic syntax tree: numbers, + - * / // % **, raiz(), abs(), pi... nothing else."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ValueError("potencia demasiado grande")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("división por cero")
        value = _BINARY[type(node.op)](left, right)
        if isinstance(value, complex) or abs(value) > MAX_MAGNITUDE:
            raise ValueError("resultado fuera de rango")
        return value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        value = _evaluate(node.args[0])
        if node.func.id in ("raiz", "sqrt") and value < 0:
            raise ValueError("raíz de un número negativo")
        return _FUNCTIONS[node.func.id](value)
    raise ValueError("expresión no válida")


def format_number(value: float) -> str:
    """Spanish style, as the model should write it: 408, 3,5, 0,333333."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def calculate(expression: str) -> str:
    """The value of an arithmetic expression, or 'error: …'."""
    text = _normalize_expression(str(expression))
    if not text or len(text) > MAX_EXPRESSION:
        return "error: expresión vacía o demasiado larga"
    try:
        return format_number(_evaluate(ast.parse(text, mode="eval")))
    except (SyntaxError, ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
        message = error.msg if isinstance(error, SyntaxError) else str(error)
        return f"error: {'expresión no válida' if isinstance(error, SyntaxError) else message}"


# ---------------------------------------------------------------- fecha_hora

DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]  # fmt: skip


def format_datetime(moment: datetime) -> str:
    """'sábado, 7 de noviembre de 2026, 14:52'."""
    day = DAYS[moment.weekday()]
    return f"{day}, {moment.day} de {MONTHS[moment.month - 1]} de {moment.year}, {moment:%H:%M}"


def now_text(_: dict | None = None) -> str:
    return format_datetime(datetime.now())


# ---------------------------------------------------------------- buscar_documentos

STOPWORDS = set(
    "a al algo como con de del el en es esa ese eso esta este esto la las le lo los mas me mi muy no o "
    "para pero por que qué se si sin sobre su sus te tu un una uno unos y ya dime dice cual cuál cuando "
    "donde quien quién hay son fue era han ser busca buscar documentos documento mis según".split()
)


def words(text: str) -> list[str]:
    """Lowercase words without accents, for matching: 'Fotosíntesis' -> 'fotosintesis'."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-zñ0-9]+", plain) if len(w) > 1 and w not in STOPWORDS]


class DocumentIndex:
    """BM25 over the paragraphs of the .md and .txt files under a folder.

    Each paragraph is a passage, prefixed by the title of its document (its first line, or the
    file name) so that a question about "Lima" also finds paragraphs that only say "la ciudad".
    """

    def __init__(self, folder: Path, k1: float = 1.5, b: float = 0.75) -> None:
        self.folder, self.k1, self.b = folder, k1, b
        self.passages: list[tuple[str, str]] = []  # (document title, paragraph)
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".md", ".txt")) if folder.exists() else []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.strip().splitlines()
            title = lines[0].lstrip("# ").strip() if lines else path.stem
            for paragraph in re.split(r"\n\s*\n", text):
                paragraph = " ".join(paragraph.split())
                # Prose only: headings, lists of links and category lines have no sentence ending.
                if len(paragraph) > 40 and not paragraph.startswith("#") and re.search(r"[.!?]", paragraph):
                    self.passages.append((title, paragraph))
        terms = [Counter(words(f"{title} {paragraph}")) for title, paragraph in self.passages]
        self.lengths = [sum(t.values()) for t in terms]
        self.average = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        # Inverted index: for each word, the passages that contain it and how often.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, counts in enumerate(terms):
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        n = len(terms)
        self.idf = {term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()}

    def search(self, query: str, k: int = 2) -> list[tuple[float, str, str]]:
        """The k best (score, title, paragraph) for the query, best first; only those that match."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(words(query)):
            idf = self.idf.get(term, 0.0)
            for i, tf in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        best = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        return [(score, *self.passages[i]) for i, score in best]


_indexes: dict[Path, tuple[tuple, DocumentIndex]] = {}


def document_index(folder: Path = DOCS_DIR) -> DocumentIndex:
    """The index of a folder, rebuilt only when its files change."""
    files = sorted(folder.rglob("*")) if folder.exists() else []
    signature = tuple((str(p), p.stat().st_mtime_ns) for p in files if p.is_file())
    cached = _indexes.get(folder)
    if cached is None or cached[0] != signature:
        _indexes[folder] = (signature, DocumentIndex(folder))
    return _indexes[folder][1]


def search_documents(query: str, folder: Path = DOCS_DIR) -> str:
    query = str(query).strip()
    if not query:
        return "error: la consulta está vacía"
    index = document_index(folder)
    if not index.passages:
        return "No hay documentos: añade archivos .md o .txt en data/docs/."
    hits = index.search(query)
    if not hits:
        return f"No se encontró nada sobre «{query}» en los documentos."
    per_hit = MAX_RESULT_CHARS // len(hits)
    parts = []
    for _, title, paragraph in hits:
        text = paragraph if len(paragraph) <= per_hit else paragraph[:per_hit].rsplit(" ", 1)[0] + "…"
        parts.append(f"[{title}] {text}")
    return "\n".join(parts)


# ---------------------------------------------------------------- registry

TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            "calculadora",
            "calcula una expresión aritmética, como 17*24 o (3+4)/2",
            {"expresion": "la expresión"},
            lambda arguments: calculate(arguments["expresion"]),
        ),
        Tool("fecha_hora", "la fecha y la hora actuales", {}, now_text),
        Tool(
            "buscar_documentos",
            "busca en los documentos del usuario",
            {"consulta": "qué buscar"},
            lambda arguments: search_documents(arguments["consulta"]),
        ),
    )
}


def system_prompt(tools: list[Tool] | None = None, base: str = DEFAULT_SYSTEM) -> str:
    """The system prompt that tells the model which tools it has (one line per tool)."""
    tools = list(TOOLS.values()) if tools is None else tools
    lines = [f"- {tool.signature}: {tool.description}" for tool in tools]
    return f"{base}\nPuedes usar estas herramientas:\n" + "\n".join(lines)
