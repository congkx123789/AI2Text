from __future__ import annotations


def run(code: str) -> str:
# Toy sandbox (no imports, eval only). Extend with care.
    allowed_builtins = {"abs": abs, "sum": sum, "min": min, "max": max}
    env = {"__builtins__": allowed_builtins}
    return str(eval(code, env, {}))