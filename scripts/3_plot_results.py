import sys
import os
import shutil
import stat
import time

# Añadir directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import config

# --- LIMPIEZA ROBUSTA ---
def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    try: func(path)
    except Exception as e: print(f"    [Advertencia] {e}")

def force_clean_directory(directory):
    if not directory.exists(): return
    print(f"(!) Limpiando carpeta antigua: {directory}")
    for i in range(3):
        try:
            shutil.rmtree(directory, onerror=on_rm_error)
            break 
        except Exception:
            time.sleep(0.5)

# --- CARGA DINÁMICA DE NSGA-II ---

def load_front_nsga(budget, seed, n_pop):
    """
    Carga frente NSGA-II basado en N y Budget -> Carpeta P{N}G{G}
    """
    generations = int(budget / n_pop)
    nsga_folder = f"P{n_pop}G{generations}"
    
    base_path = config.NSGA_DIR / f"ZDT3_{budget}" / nsga_folder
    pattern = f"*seed{seed:02d}.out"
    found = list(base_path.glob(pattern))
    
    if not found: return None
    
    try:
        data = np.loadtxt(found[0])
        if data.ndim == 1: return data[0:2].reshape(1, 2)
        return data[:, 0:2] # Cols 0 y 1 son objetivos
    except Exception:
        return None

def plot_comparison(budget, exp_name, seed, params):
    # Cargar MOEA/D
    moea_file = config.RESULTS_DIR / exp_name / str(budget) / f"seed_{seed:02d}.txt"
    if not moea_file.exists(): return

    try:
        F_moea = np.loadtxt(moea_file)[:, -2:] 
    except: return

    # Cargar NSGA-II Dinámico (usando N del experimento)
    n_pop = params['N']
    F_nsga = load_front_nsga(budget, seed, n_pop)
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    # Fondo: NSGA-II (si existe)
    if F_nsga is not None:
        label_nsga = f'NSGA-II (N={n_pop})'
        plt.scatter(F_nsga[:, 0], F_nsga[:, 1], c='orange', marker='x', alpha=0.5, label=label_nsga)
    else:
        # Si no tienes esa carpeta de NSGA, no rompe, solo avisa visualmente en el plot si quieres
        pass 
    
    # Frente: Tu Algoritmo
    plt.scatter(F_moea[:, 0], F_moea[:, 1], c='dodgerblue', s=30, edgecolors='k', linewidth=0.5, label=f'MOEA/D ({exp_name})')
    
    plt.title(f"ZDT3: {exp_name} vs NSGA-II\n(Eval: {budget}, N: {n_pop}, Seed: {seed})")
    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0.0, 1.1)
    plt.ylim(-0.8, 1.2)
    
    out_name = config.FIGURES_DIR / f"Compare_{exp_name}_B{budget}_S{seed:02d}.png"
    plt.savefig(out_name, dpi=150)
    plt.close()
    print(f"   Generado: {out_name.name}")

def main():
    print("--- GENERANDO GRÁFICAS COMPARATIVAS ---")
    
    # Limpieza inicial
    force_clean_directory(config.FIGURES_DIR)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    SEED_TO_PLOT = 1 
    
    for exp_name, params in config.EXPERIMENTS.items():
        print(f">>> Gráficas para: {exp_name}")
        for budget in config.BUDGETS:
            plot_comparison(budget, exp_name, SEED_TO_PLOT, params)
            
    print("\n[OK] Todas las gráficas generadas.")

if __name__ == "__main__":
    main()