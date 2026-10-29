"""Tools the reasoning agent can call via function calling."""

from __future__ import annotations

import ast
import operator as op
from collections.abc import Callable
from typing import Any

_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without ``eval``. Rejects names, calls and attribute access."""

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(_ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")

    if len(expression) > 200:
        raise ValueError("expression too long")
    return _ev(ast.parse(expression, mode="eval"))


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '(12.5 * 4) / 3'.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the document store and return the most relevant passages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


class ToolRegistry:
    def __init__(self, search_fn: Callable[[str], str] | None = None) -> None:
        self._fns: dict[str, Callable[..., str]] = {"calculator": self._calculator}
        self.schemas = [CALCULATOR_SCHEMA]
        if search_fn is not None:
            self._fns["search_docs"] = lambda query: search_fn(query)
            self.schemas.append(SEARCH_SCHEMA)

    @staticmethod
    def _calculator(expression: str) -> str:
        result = safe_eval(expression)
        return str(int(result)) if float(result).is_integer() else f"{result:.6g}"

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        fn = self._fns.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}"
        try:
            return fn(**arguments)
        except Exception as exc:  # tool errors go back to the model, never crash the run
            return f"error: {exc}"
