# src/problem.py
import numpy as np

class ZDT3:
    def __init__(self, dim=30):
        self.dim = dim
        self.lower = np.zeros(dim)
        self.upper = np.ones(dim)
        self.n_objs = 2

    def evaluate(self, x):
        """Evalúa un vector x y devuelve [f1, f2]."""
        # Asegurar límites antes de evaluar
        x = np.clip(x, self.lower, self.upper)
        
        f1 = x[0]
        
        # g(x)
        g = 1 + 9 * np.mean(x[1:])
        
        # h(f1, g) para ZDT3 (discontinuo)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        
        f2 = g * h
        return np.array([f1, f2])

    def random_solution(self, rng):
        """Genera una solución aleatoria dentro de los límites."""
        return rng.uniform(self.lower, self.upper)