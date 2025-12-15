import os
import sys

# Añadimos el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import config
from src.algorithm import MOEAD


def save_results(path, pop, objs):
    """Guarda población y objetivos: [x1...x30 f1 f2]"""
    data = np.hstack((pop, objs))
    np.savetxt(path, data, fmt='%.6e', delimiter='\t')

def main():
    print(f"--- INICIANDO EJECUCIÓN DE EXPERIMENTOS ---")
    
    # Aseguramos que el directorio base exista, pero SIN borrar lo que haya dentro
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directorio base listo (acumulativo): {config.RESULTS_DIR}\n")
    
    # --- EJECUCIÓN ---
    for exp_name, params in config.EXPERIMENTS.items():
        print(f">>> Ejecutando: {exp_name}")
        
        for budget in config.BUDGETS:
            output_folder = config.RESULTS_DIR / exp_name / str(budget)
            # Crea la carpeta específica del experimento si no existe
            output_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"   [Presupuesto: {budget}] -> {output_folder}")
            
            for seed in config.SEEDS:
                filename = output_folder / f"seed_{seed:02d}.txt"
                try:
                    moea = MOEAD(params, seed)
                    final_pop, final_objs = moea.run(budget)
                    save_results(filename, final_pop, final_objs)
                except Exception as e:
                    print(f"     [ERROR] Seed {seed:02d}: {e}")
            
            print(f"     -> 10 Semillas OK.")

    print("\n--- FINALIZADO ---")
    print("Siguiente paso: python scripts/2_compute_metrics.py")

if __name__ == "__main__":
    main()