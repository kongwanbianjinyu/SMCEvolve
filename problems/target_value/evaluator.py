"""Toy problem: drive `result` toward 42.0.

reward = 1 / (1 + |result - 42|), so the optimum is 1.0 at result == 42 and
any invalid program gets 0.0.
"""

TARGET = 42.0


def evaluate(program: str) -> float:
    namespace: dict = {}
    try:
        exec(program, namespace)
    except Exception:
        return 0.0
    value = namespace.get("result")
    if not isinstance(value, (int, float)):
        return 0.0
    return 1.0 / (1.0 + abs(float(value) - TARGET))
