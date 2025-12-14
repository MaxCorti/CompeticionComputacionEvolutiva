# scripts/3_plot_results.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import config

def load_front_nsga(budget, seed):
    """Carga frente NSGA-II detectando formato (35 cols vs 2 cols)."""
    nsga_folder_name = config.NSGA_MAPPING.get(budget)
    if not nsga_folder_name: return None
    
    # Construir ruta base: data/nsga2_reference/ZDT3_4000/P100G40/
    base_path = config.NSGA_DIR / f"ZDT3_{budget}" / nsga_folder_name
    
    # Buscar fichero por patrón
    pattern = f"*seed{seed:02d}.out"
    found = list(base_path.glob(pattern))
    
    if not found: return None
    
    data = np.loadtxt(found[0])
    
    # Detectar formato
    if data.shape[1] >= 32:
        # Formato jMetal/Standard: 30 vars + 2 objs + ...
        # Objetivos en columnas 30 y 31 (indices)
        return data[:, 30:32]
    else:
        # Formato simple (si lo hubiera): solo objetivos
        return data[:, 0:2]

def plot_comparison(budget, exp_name, seed):
    # Rutas
    moea_file = config.RESULTS_DIR / exp_name / str(budget) / f"seed_{seed:02d}.txt"
    
    if not moea_file.exists():
        print(f"[SKIP] No hay datos para {exp_name} B={budget} S={seed}")
        return

    # Cargar datos
    F_moea = np.loadtxt(moea_file)[:, -2:] # Últimas 2 columnas
    F_nsga = load_front_nsga(budget, seed)
    
    # Crear Figura
    plt.figure(figsize=(8, 6))
    
    # Frente NSGA-II (Fondo)
    if F_nsga is not None:
        plt.scatter(F_nsga[:, 0], F_nsga[:, 1], c='orange', marker='x', alpha=0.6, label='NSGA-II (Referencia)')
    
    # Frente MOEA/D (Primer plano)
    plt.scatter(F_moea[:, 0], F_moea[:, 1], c='dodgerblue', s=30, edgecolors='k', linewidth=0.5, label=f'MOEA/D ({exp_name})')
    
    plt.title(f"Comparativa ZDT3 - {exp_name}\nPresupuesto: {budget} - Seed: {seed}")
    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1.1)
    plt.ylim(-0.8, 1.2) # Ajuste visual ZDT3
    
    # Guardar
    out_dir = config.FIGURES_DIR
    out_dir.mkdir(exist_ok=True)
    out_name = out_dir / f"Compare_{exp_name}_B{budget}_S{seed:02d}.png"
    
    plt.savefig(out_name, dpi=150)
    plt.close()
    print(f"Generado: {out_name}")

def main():
    print("--- GENERANDO GRÁFICAS COMPARATIVAS ---")
    
    # Generar para Seed 1 como muestra (o recorrer todas si quieres)
    SEED_TO_PLOT = 1
    
    for exp_name in config.EXPERIMENTS.keys():
        for budget in config.BUDGETS:
            plot_comparison(budget, exp_name, SEED_TO_PLOT)

if __name__ == "__main__":
    main()