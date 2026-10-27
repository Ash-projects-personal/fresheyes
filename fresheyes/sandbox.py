"""The Python sandbox for re-running an agent's computations.

Only arithmetic is allowed: numbers and + - * / // %. No names, no calls,
no attribute access, no imports, no indexing. The expression is parsed with
``ast`` and the tree is walked by hand; anything outside the allowed node
set is rejected before any evaluation happens. A computation that cannot be
parsed or that divides by zero is reported as an error, never guessed at.
"""

from __future__ import annotations

import ast

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def evaluate(expression: str) -> float:
    """Evaluate an arithmetic expression string. Raises ValueError if the
    expression uses anything but plain arithmetic on numbers."""
    # Commas are thousands separators in written numbers, never valid
    # arithmetic, so strip them before parsing ("8,214" is one number).
    expression = expression.replace(",", "")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse calculation: {exc}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ValueError("division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ValueError("division by zero")
            return left % right
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError(f"not plain arithmetic: {type(node).__name__}")


def formula_inputs(expression: str) -> list[float]:
    """The numbers a formula is built from, so the checker can verify each
    input against the cited source document."""
    expression = expression.replace(",", "")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return []
    inputs: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            inputs.append(float(node.value))
    return inputs
