# main.py
import numpy as np

from problems import create_zdt3_problem
from de_aggregation import aggregation_moea


def save_population(filename: str, pop: np.ndarray, objs: np.ndarray):
    """
    Guarda en un fichero de texto:
    x_1 ... x_p  f_1  f_2
    una solución por línea.
    """
    data = np.hstack([pop, objs])
    np.savetxt(filename, data, fmt="%.6e")


def main():
    # --- Parámetros del problema ZDT3 ---
    dim = 30
    lower, upper, obj_func = create_zdt3_problem(dim)

    # --- Parámetros del algoritmo basado en agregación ---
    N = 100          # número de subproblemas / tamaño población
    G = 100          # número de generaciones  -> N * G = 10000 evaluaciones aprox
    T = 20           # tamaño de vecindad (20% de N)
    F = 0.5          # parámetro DE
    CR = 0.5         # cruce DE
    seed = 1234      # semilla para reproducibilidad

    # Si quieres fijar exactamente el presupuesto:
    max_evals = N * G  # opcional; podrías poner, por ejemplo, 10000 o 4000

    print("Ejecutando algoritmo de agregación sobre ZDT3...")
    pop, objs, z, evals = aggregation_moea(
        N=N,
        G=G,
        T=T,
        lower=lower,
        upper=upper,
        obj_func=obj_func,
        F=F,
        CR=CR,
        seed=seed,
        max_evals=max_evals,
    )

    print(f"Evaluaciones realizadas: {evals}")
    print(f"Punto de referencia z*: {z}")

    # Guardar población final y valores objetivo
    filename = f"final_pop_ZDT3_N{N}_G{G}.txt"
    save_population(filename, pop, objs)
    print(f"Población final guardada en: {filename}")


if __name__ == "__main__":
    main()
