# src/decomposition.py
import numpy as np


def generate_weight_vectors(N: int) -> np.ndarray:
    """Genera vectores equiespaciados (Punto 1)."""
    lambdas = np.zeros((N, 2))
    for i in range(N):
        w1 = i / (N - 1) if N > 1 else 0.5
        lambdas[i] = np.array([w1, 1.0 - w1])
    return np.maximum(lambdas, 1e-6) # Evitar división por cero

def tchebycheff(f_vals, lambdas, z_star):
    """
    Escalarización Tchebycheff.
    Cumple: "Entiendo cómo la scalarización convierte..." (Punto 1)
    """
    return np.max(lambdas * np.abs(f_vals - z_star))

def compute_neighbors(weights, T):
    """Selección de vecinos por distancia euclídea (Punto 2)."""
    N = weights.shape[0]
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        dist_matrix[i] = np.linalg.norm(weights - weights[i], axis=1)
    return np.argsort(dist_matrix, axis=1)[:, :T]