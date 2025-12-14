# de_aggregation.py
import numpy as np


def generate_weight_vectors(N: int, m: int) -> np.ndarray:
    """
    Genera N vectores de peso para m objetivos.
    Para m = 2, se generan equiespaciados en la línea λ1 + λ2 = 1.
    Para este trabajo solo necesitamos m = 2 (ZDT3).
    """
    if m != 2:
        raise NotImplementedError("Esta implementación simple solo soporta m = 2 objetivos.")
    lambdas = np.zeros((N, 2))
    if N == 1:
        lambdas[0] = np.array([0.5, 0.5])
        return lambdas

    for i in range(N):
        w1 = i / (N - 1)
        w2 = 1.0 - w1
        lambdas[i] = np.array([w1, w2])
    return lambdas


def compute_neighbors(weights: np.ndarray, T: int) -> np.ndarray:
    """
    Calcula los índices de los T vecinos más cercanos (distancia euclídea) para cada vector de pesos.
    Devuelve una matriz de tamaño (N, T) con índices.
    """
    N = weights.shape[0]
    T = min(max(T, 1), N)
    # Distancias cuadradas
    diff = weights[:, None, :] - weights[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    neighbors = np.argsort(dist2, axis=1)[:, :T]
    return neighbors


def tchebycheff(Fx: np.ndarray, lambd: np.ndarray, z: np.ndarray) -> float:
    """
    Scalarización de Tchebycheff:
    g(x | λ, z*) = max_j λ_j * |f_j(x) - z*_j|
    """
    return float(np.max(lambd * np.abs(Fx - z)))


def differential_evolution_and_gaussian_mutation(
    pop: np.ndarray,
    i: int,
    neighbors: np.ndarray,
    F: float,
    CR: float,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    SIG: float = 20.0,
) -> np.ndarray:
    """
    Operador de reproducción:
    - Usamos Evolución Diferencial (mutación + cruce binomial) restringida a la vecindad.
    - Luego aplicamos mutación gaussiana con probabilidad 1/p por variable.
    """
    N, p = pop.shape
    # Índices de los vecinos de i (incluyendo i)
    B_i = neighbors[i]
    # Elegimos 3 índices distintos de la vecindad
    if len(B_i) < 3:
        # Si la vecindad es demasiado pequeña, usamos toda la población
        B_i = np.arange(N)

    r1, r2, r3 = rng.choice(B_i, size=3, replace=False)

    x_i = pop[i]
    x_r1 = pop[r1]
    x_r2 = pop[r2]
    x_r3 = pop[r3]

    # Mutación DE: v = x_r1 + F * (x_r2 - x_r3)
    v = x_r1 + F * (x_r2 - x_r3)

    # Cruce binomial: u_j = v_j con prob CR o para una dimensión j_rand obligatoria
    u = np.copy(x_i)
    j_rand = rng.integers(p)
    for j in range(p):
        if rng.random() < CR or j == j_rand:
            u[j] = v[j]

    # Mutación gaussiana con probabilidad 1/p
    sigma = (upper - lower) / SIG
    for j in range(p):
        if rng.random() < 1.0 / p:
            # np.random.Generator.normal ya genera N(0,1)
            u[j] += rng.normal(loc=0.0, scale=sigma[j])

    # Reparación de límites
    u = np.clip(u, lower, upper)
    return u


def aggregation_moea(
    N: int,
    G: int,
    T: int,
    lower: np.ndarray,
    upper: np.ndarray,
    obj_func,
    F: float = 0.5,
    CR: float = 0.5,
    seed: int | None = None,
    max_evals: int | None = None,
):
    """
    Implementación básica del algoritmo multiobjetivo basado en agregación (Tchebycheff):

    - N: número de subproblemas (tamaño de población)
    - G: número de generaciones
    - T: tamaño de vecindad
    - lower, upper: arrays con límites inferior y superior por variable
    - obj_func: función f(x) que devuelve vector de objetivos (np.ndarray de tamaño m)
    - F, CR: parámetros de DE
    - seed: semilla para reproducibilidad
    - max_evals: límite opcional de evaluaciones (si None, se asume N * G)

    Devuelve:
    - pop: población final (N x p)
    - objs: objetivos finales (N x m)
    - z: punto de referencia (vector de tamaño m)
    - evals: número total de evaluaciones realizadas
    """
    rng = np.random.default_rng(seed)

    p = lower.shape[0]
    # Inicializamos población aleatoria
    pop = rng.uniform(low=lower, high=upper, size=(N, p))

    # Evaluamos población inicial
    objs = np.array([obj_func(ind) for ind in pop])
    evals = N

    # Número de objetivos
    m = objs.shape[1]

    # Generamos vectores de peso y vecindades
    weights = generate_weight_vectors(N, m)
    neighbors = compute_neighbors(weights, T)

    # Punto de referencia z: mejores valores por objetivo
    z = np.min(objs, axis=0)

    # Bucle principal de generaciones
    for g in range(G):
        for i in range(N):
            if max_evals is not None and evals >= max_evals:
                # Hemos alcanzado el presupuesto de evaluaciones
                return pop, objs, z, evals

            # 1) Reproducción: DE + mutación gaussiana
            y = differential_evolution_and_gaussian_mutation(
                pop=pop,
                i=i,
                neighbors=neighbors,
                F=F,
                CR=CR,
                lower=lower,
                upper=upper,
                rng=rng,
            )

            # 2) Evaluación
            Fy = obj_func(y)
            evals += 1

            # 3) Actualización del punto de referencia z
            z = np.minimum(z, Fy)

            # 4) Actualización de vecinos
            Bi = neighbors[i]
            for j in Bi:
                gj_old = tchebycheff(objs[j], weights[j], z)
                gj_new = tchebycheff(Fy,       weights[j], z)
                if gj_new <= gj_old:
                    pop[j] = y
                    objs[j] = Fy

    return pop, objs, z, evals
