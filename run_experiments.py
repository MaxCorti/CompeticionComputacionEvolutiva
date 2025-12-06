# run_experiments.py
import numpy as np
from pathlib import Path

from problems import create_zdt3_problem
from de_aggregation import aggregation_moea


# Configuraciones de experimento
CONFIGS = [
    {
        "name": "ZDT3_10000",
        "N": 100,
        "G": 100,   # N * G = 10 000 evals
        "T": 20,    # 20% de N
        "budget": 10000,
    },
    {
        "name": "ZDT3_4000",
        "N": 100,
        "G": 40,    # N * G = 4 000 evals
        "T": 20,
        "budget": 4000,
    },
]

SEEDS = list(range(1, 11))  # 10 ejecuciones por condición
BASE_DIR = Path("results") / "ZDT3"


def save_population(filename: Path, pop: np.ndarray, objs: np.ndarray) -> None:
    """
    Guarda en un fichero de texto:
    x_1 ... x_p  f_1  f_2
    una solución por línea.
    """
    data = np.hstack([pop, objs])
    np.savetxt(filename, data, fmt="%.6e")


def main():
    # Definir problema ZDT3-30
    dim = 30
    lower, upper, obj_func = create_zdt3_problem(dim=dim)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        N = cfg["N"]
        G = cfg["G"]
        T = cfg["T"]
        budget = cfg["budget"]

        subdir = BASE_DIR / f"P{N}G{G}"
        subdir.mkdir(parents=True, exist_ok=True)

        log_path = subdir / "log_ejecuciones.txt"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(
                f"Configuración: {cfg['name']}\n"
                f"N = {N}, G = {G}, T = {T}, presupuesto = {budget} evals\n"
                f"Dimensión = {dim}\n\n"
            )

            for seed in SEEDS:
                print(f"\n=== Ejecutando {cfg['name']} - seed {seed} ===")
                pop, objs, z, evals = aggregation_moea(
                    N=N,
                    G=G,
                    T=T,
                    lower=lower,
                    upper=upper,
                    obj_func=obj_func,
                    F=0.5,
                    CR=0.5,
                    seed=seed,
                    max_evals=budget,
                )

                # Guardar población final
                filename = subdir / f"final_pop_seed{seed:02d}.txt"
                save_population(filename, pop, objs)

                # Log básico
                log_file.write(
                    f"seed={seed:02d}  -> evals={evals}, "
                    f"z*={z}, fichero={filename.name}\n"
                )

                print(f"Guardado: {filename} (evals={evals}, z*={z})")

        print(f"\nTodos los experimentos para {cfg['name']} guardados en {subdir}\n")


if __name__ == "__main__":
    main()
