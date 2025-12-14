# src/operators.py
import numpy as np

def differential_evolution_op(current_idx, population, neighbors, F, CR, rng):
    """Genera un individuo mutado usando DE/rand/1/bin sobre vecindad."""
    
    # Seleccionar 3 padres distintos de la vecindad
    # Si la vecindad es muy pequeña (<3), usamos toda la población (fallback)
    pool = neighbors if len(neighbors) >= 3 else np.arange(len(population))
    
    r1, r2, r3 = rng.choice(pool, size=3, replace=False)
    
    x_r1 = population[r1]
    x_r2 = population[r2]
    x_r3 = population[r3]
    x_target = population[current_idx]
    
    # Mutación
    v = x_r1 + F * (x_r2 - x_r3)
    
    # Cruce Binomial
    dim = len(x_target)
    j_rand = rng.integers(0, dim)
    
    u = np.copy(x_target)
    for j in range(dim):
        if rng.random() < CR or j == j_rand:
            u[j] = v[j]
            
    return u

def polynomial_mutation(x, lower, upper, pm, eta_m, rng):
    """Mutación polinómica (estándar en NSGA-II), alternativa a Gaussiana."""
    # Implementación opcional si quieres comparar operadores
    pass

def gaussian_mutation(x, lower, upper, pm, sig, rng):
    """Mutación Gaussiana simple."""
    u = np.copy(x)
    dim = len(x)
    sigma = (upper - lower) / sig
    
    for j in range(dim):
        if rng.random() < pm:
            u[j] += rng.normal(0, sigma[j])
            
    return np.clip(u, lower, upper)