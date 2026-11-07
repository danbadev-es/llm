"""Checks for leon.tools and leon.agent (phase 6). Run: pytest tests/test_agent.py"""

import tempfile
from datetime import datetime
from pathlib import Path

from leon.agent import parse_call, run_agent
from leon.chat import encode_chat
from leon.tokenizer import BPETokenizer, count_chunks
from leon.tools import TOOLS, DocumentIndex, Tool, calculate, format_datetime, search_documents, system_prompt

TOKENIZER = BPETokenizer.train(count_chunks(['hola {"name": "calculadora", "arguments": 17*24} son 408'] * 5), 400)
S = TOKENIZER.special_tokens


def test_calculator_is_exact_and_safe() -> None:
    assert calculate("17*24") == "408"
    assert calculate("(3+4)/2") == "3,5"  # Spanish decimal comma, as the model must write it
    assert calculate("3,5 * 2") == "7"
    assert calculate("2^10") == "1024"
    assert calculate("raiz(81) + 1") == "10"
    assert calculate("1/0") == "error: división por cero"
    for attack in ('__import__("os").system("x")', "open('f')", "(1).__class__", "9**9**9", "a" * 500):
        assert calculate(attack).startswith("error:"), attack


def test_dates_are_written_in_spanish() -> None:
    assert format_datetime(datetime(2026, 11, 7, 14, 5)) == "sábado, 7 de noviembre de 2026, 14:05"


def test_document_search_finds_the_right_passage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "lima.md").write_text(
            "# Lima\n\nLima es la capital del Perú y su ciudad más poblada, a orillas del Pacífico.\n",
            encoding="utf-8",
        )
        (folder / "fotosintesis.txt").write_text(
            "Fotosíntesis\n\nLa fotosíntesis convierte la luz del sol en energía química en las plantas.\n",
            encoding="utf-8",
        )
        index = DocumentIndex(folder)
        assert index.search("¿Cuál es la capital de Perú?")[0][1] == "Lima"
        assert index.search("plantas y luz solar")[0][1] == "Fotosíntesis"  # accents do not matter
        assert search_documents("fotosintesis", folder).startswith("[Fotosíntesis] La fotosíntesis")
        assert "No se encontró" in search_documents("ajedrez", folder)
        assert "No hay documentos" in search_documents("x", folder / "vacía")


def test_tools_never_raise_and_report_missing_arguments() -> None:
    assert TOOLS["calculadora"]({}) == "error: falta el argumento expresion"
    assert TOOLS["calculadora"]("17") == "error: los argumentos deben ser un objeto JSON"
    broken = Tool("roto", "falla", {}, lambda _: 1 / 0)
    assert broken({}).startswith("error:")
    assert system_prompt().count("\n- ") == len(TOOLS)


def test_parse_call() -> None:
    assert parse_call('{"name": "fecha_hora", "arguments": {}}') == {"name": "fecha_hora", "arguments": {}}
    assert parse_call('{"name": "fecha_hora"}')["arguments"] == {}
    for bad in ("{", '{"arguments": {}}', '{"name": "x", "arguments": 3}', "[1]"):
        try:
            parse_call(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def scripted(*segments: list[int]):
    """A fake model: each call to generate() replays the next segment of token ids."""
    calls = iter(segments)
    prompts = []

    def generate(ids, max_new, stop_ids):
        prompts.append(list(ids))
        for token in next(calls)[:max_new]:
            yield token
            if token in stop_ids:
                return

    return generate, prompts


def call_tokens(text: str) -> list[int]:
    return [S["<|tool_call|>"], *TOKENIZER.encode(text), S["<|/tool_call|>"]]


def test_agent_runs_the_tool_and_continues_the_same_turn() -> None:
    prompt = encode_chat(TOKENIZER, [{"role": "user", "content": "hola"}])
    generate, prompts = scripted(
        call_tokens('{"name": "calculadora", "arguments": {"expresion": "17*24"}}'),
        [*TOKENIZER.encode("son 408"), S["<|end|>"]],
    )
    events = list(run_agent(generate, TOKENIZER, prompt, max_new_tokens=200))
    assert events[0] == {"tool_call": {"name": "calculadora", "arguments": {"expresion": "17*24"}}}
    assert events[1] == {"tool_result": "408"}
    assert "".join(e.get("text", "") for e in events) == "son 408"
    assert events[-1]["stopped"] == "end" and events[-1]["tool_calls"] == 1
    # The second generation sees the call and its result, and no new <|assistant|>.
    assert TOKENIZER.decode(prompts[1]).endswith('"17*24"}}<|/tool_call|><|tool_result|>408<|/tool_result|>')


def test_agent_reports_invalid_calls_to_the_model_and_limits_calls() -> None:
    prompt = encode_chat(TOKENIZER, [{"role": "user", "content": "hola"}])
    generate, prompts = scripted(call_tokens("{roto"), [*TOKENIZER.encode("hola"), S["<|end|>"]])
    events = list(run_agent(generate, TOKENIZER, prompt, max_new_tokens=200))
    assert events[0]["tool_call"]["error"] == "la llamada no es JSON válido"
    assert events[1] == {"tool_result": "error: la llamada no es JSON válido"}
    assert events[-1]["stopped"] == "end"

    call = call_tokens('{"name": "fecha_hora", "arguments": {}}')
    generate, _ = scripted(call, call, call)
    events = list(run_agent(generate, TOKENIZER, prompt, max_new_tokens=500, max_calls=2))
    assert len([e for e in events if "tool_call" in e]) == 2
    assert events[-1]["stopped"] == "max_tool_calls"


def test_agent_stops_at_the_token_budget() -> None:
    prompt = encode_chat(TOKENIZER, [{"role": "user", "content": "hola"}])
    generate, _ = scripted(TOKENIZER.encode("hola hola hola hola"))
    events = list(run_agent(generate, TOKENIZER, prompt, max_new_tokens=3))
    assert events[-1] == {"done": True, "stopped": "max_tokens", "tokens": 3, "tool_calls": 0}
