# utils/validators.py
from __future__ import annotations

def normalize_axes(x: str, y: str, z: str):
    """
    Regras:
    - X e Y: obrigatórios -> se 'Qualquer', substitui por defaults.
    - Z: opcional -> pode ser 'Qualquer'.
    """
    default_x = "temperature"
    default_y = "age"

    changed = False
    messages = []

    if x == "Qualquer":
        x = default_x
        changed = True
        messages.append(f"Eixo X estava 'Qualquer' e foi ajustado para '{default_x}'.")

    if y == "Qualquer":
        if default_y == x:
            default_y = "income" if x != "income" else "temperature"
        y = default_y
        changed = True
        messages.append(f"Eixo Y estava 'Qualquer' e foi ajustado para '{default_y}'.")

    return x, y, z, changed, messages


def ensure_min_samples(n: int, minimum: int = 50):
    if n < minimum:
        raise ValueError(f"Amostras insuficientes para análise confiável (mínimo: {minimum}).")
