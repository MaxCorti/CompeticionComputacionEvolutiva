# problems.py
import numpy as np

def zdt3(x: np.ndarray) -> np.ndarray:
    """
    Problema ZDT3 estándar.
    x: vector de decisión (dimensión n >= 2) con valores en [0, 1].
    Devuelve np.array([f1, f2]).
    """
    x = np.asarray(x)
    n = x.shape[0]
    f1 = x[0]
    g = 1.0 + 9.0 / (n - 1) * np.sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
    f2 = g * h
    return np.array([f1, f2])


def create_zdt3_problem(dim: int = 30):
    """
    Devuelve límites inferior/superior y la función objetivo para ZDT3-dim.
    """
    lower = np.zeros(dim)
    upper = np.ones(dim)
    return lower, upper, zdt3
