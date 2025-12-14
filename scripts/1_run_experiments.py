# scripts/1_run_experiments.py
import os
import sys

# Añadir directorio raíz al path para importar src y config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import config
from src.algorithm import MOEAD


def save_results(path, pop, objs):
    """Guarda población y objetivos en un fichero de texto."""
    # Concatenar variables y objetivos: [x1...x30, f1, f2]
    data = np.hstack((pop, objs))
    np.savetxt(path, data, fmt='%.6e')

def main():
    print(f"--- INICIANDO EJECUCIÓN MASIVA DE EXPERIMENTOS ---")
    print(f"Salida de datos: {config.RESULTS_DIR}\n")
    
    # Recorrer experimentos definidos en config.py
    for exp_name, params in config.EXPERIMENTS.items():
        print(f">>> Experimento: {exp_name}")
        
        # Recorrer presupuestos (4000, 10000)
        for budget in config.BUDGETS:
            # Crear estructura de carpetas: experiments/BASELINE/4000/
            out_dir = config.RESULTS_DIR / exp_name / str(budget)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"   - Presupuesto: {budget} evals")
            
            # Ejecutar N semillas
            for seed in config.SEEDS:
                # print(f"     Running Seed {seed}...", end='\r')
                
                # Instanciar y ejecutar
                moea = MOEAD(params, seed)
                pop, objs = moea.run(budget)
                
                # Guardar resultado
                filename = out_dir / f"seed_{seed:02d}.txt"
                save_results(filename, pop, objs)
            
            print(f"     [OK] 10 Semillas completadas.")
    
    print("\n--- TODOS LOS EXPERIMENTOS FINALIZADOS ---")

if __name__ == "__main__":
    main()