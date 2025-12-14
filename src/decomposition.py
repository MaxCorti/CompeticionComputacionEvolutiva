# src/decomposition.py
import numpy as np


def generate_weight_vectors(N: int) -> np.ndarray:
    """Genera N vectores de peso equiespaciados para 2 objetivos."""
    lambdas = np.zeros((N, 2))
    for i in range(N):
        if N == 1:
            w1 = 0.5
        else:
            w1 = i / (N - 1)
        w2 = 1.0 - w1
        lambdas[i] = np.array([w1, w2])
    
    # Evitar ceros absolutos para estabilidad numérica en Tchebycheff
    lambdas[lambdas < 1e-6] = 1e-6
    return lambdas

def compute_neighbors(weights: np.ndarray, T: int) -> np.ndarray:
    """Calcula los índices de los T vecinos más cercanos (Euclídea)."""
    N = weights.shape[0]
    T = min(T, N)
    dist_matrix = np.zeros((N, N))
    
    for i in range(N):
        # Distancia euclídea vectorizada
        dist = np.linalg.norm(weights - weights[i], axis=1)
        dist_matrix[i] = dist
            
    # Ordenar y coger los T primeros
    neighbors = np.argsort(dist_matrix, axis=1)[:, :T]
    return neighbors

def tchebycheff(f_vals: np.ndarray, lambdas: np.ndarray, z_star: np.ndarray) -> float:
    """Función de escalarización de Tchebycheff: max(lambda * |f - z*|)"""
    diff = np.abs(f_vals - z_star)
    return np.max(lambdas * diff)