import os
import sys

# Añadir directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

import config

# --- UTILIDADES DE CARGA ---

def load_moead_front(path):
    """Carga frentes MOEA/D [x1...xn, f1, f2]. Devuelve f1, f2."""
    if not path.exists(): return None
    try:
        data = np.loadtxt(path)
        if data.ndim == 1: return data[-2:].reshape(1, 2)
        return data[:, -2:]
    except Exception:
        return None

def load_nsga_front(budget, seed, n_pop):
    """
    Carga frentes NSGA-II calculando dinámicamente la carpeta P{N}G{G}.
    """
    # 1. Calcular Generaciones correspondientes (G = Budget / N)
    # Ejemplo: 4000 / 80 = 50 -> Busca carpeta "P80G50"
    generations = int(budget / n_pop)
    nsga_folder = f"P{n_pop}G{generations}"
    
    # 2. Construir ruta
    base_path = config.NSGA_DIR / f"ZDT3_{budget}" / nsga_folder
    pattern = f"*seed{seed:02d}.out"
    
    # 3. Buscar archivo
    found = list(base_path.glob(pattern))
    if not found: 
        # Opcional: Avisar si falta el fichero de referencia específico
        # print(f" [!] Falta referencia NSGA: {base_path}/{pattern}")
        return None
    
    try:
        data = np.loadtxt(found[0])
        # NSGA-II suele tener objetivos en cols 0 y 1
        if data.ndim == 1: return data[0:2].reshape(1, 2)
        return data[:, 0:2]
    except Exception:
        return None

def get_nondominated(front):
    """Filtra soluciones no dominadas (Minimización)."""
    if front is None or len(front) == 0: return front
    costs = front
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | \
                                         np.any(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True 
    return costs[is_efficient]

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

# --- MÉTRICAS ---

def spacing(front):
    front = get_nondominated(front)
    if front is None or len(front) < 2: return 0.0
    front = front[np.argsort(front[:, 0])]
    dists = np.sum(np.abs(front[:-1] - front[1:]), axis=1)
    d_mean = np.mean(dists)
    return np.sqrt(np.mean((dists - d_mean)**2))

def hypervolume(front, ref_point):
    front = get_nondominated(front)
    if front is None or len(front) == 0: return 0.0
    front = front[np.argsort(front[:, 0])]
    front = front[(front[:,0] <= ref_point[0]) & (front[:,1] <= ref_point[1])]
    if len(front) == 0: return 0.0
    
    hv = 0.0
    max_y = ref_point[1]
    for p in front:
        width = ref_point[0] - p[0]
        height = max_y - p[1]
        if height > 0 and width > 0:
            hv += width * height
            max_y = p[1]
    return hv

def coverage_metric(A, B):
    if B is None or len(B) == 0: return 0.0
    if A is None or len(A) == 0: return 0.0
    
    dom_count = 0
    for sol_b in B:
        for sol_a in A:
            if dominates(sol_a, sol_b):
                dom_count += 1
                break
    return dom_count / len(B)

# --- MAIN ---

def main():
    summary_data = []
    print(f"--- CALCULANDO MÉTRICAS (Ref. Point: {config.HV_REF_POINT}) ---")
    
    # Iterar sobre cada experimento configurado en config.py
    for exp_name, params in config.EXPERIMENTS.items():
        # Extraemos el tamaño de población (N) de la configuración
        n_pop = params['N']
        
        for budget in config.BUDGETS:
            hv_vals, sp_vals = [], []
            c_ab_vals, c_ba_vals = [], [] # C(MOEA, NSGA), C(NSGA, MOEA)
            
            print(f"Procesando {exp_name} (N={n_pop}) - {budget} evals...")
            
            for seed in config.SEEDS:
                # 1. Cargar MOEA/D
                path_moea = config.RESULTS_DIR / exp_name / str(budget) / f"seed_{seed:02d}.txt"
                F_moea = load_moead_front(path_moea)
                if F_moea is None: continue

                # 2. Cargar NSGA-II (Dinámico: P{N}G{Budget/N})
                F_nsga = load_nsga_front(budget, seed, n_pop)
                
                # 3. Métricas propias
                hv_vals.append(hypervolume(F_moea, np.array(config.HV_REF_POINT)))
                sp_vals.append(spacing(F_moea))
                
                # 4. Métricas comparativas (Coverage)
                if F_nsga is not None:
                    ND_moea = get_nondominated(F_moea)
                    ND_nsga = get_nondominated(F_nsga)
                    c_ab_vals.append(coverage_metric(ND_moea, ND_nsga))
                    c_ba_vals.append(coverage_metric(ND_nsga, ND_moea))
            
            if hv_vals:
                row = {
                    "Experimento": exp_name,
                    "N": n_pop,          # Dato útil para ver en la tabla
                    "Presupuesto": budget,
                    "HV (Mean)": np.mean(hv_vals),
                    "HV (Std)": np.std(hv_vals),
                    "Spacing (Mean)": np.mean(sp_vals),
                    "Spacing (Std)": np.std(sp_vals),
                    "C(Alg, NSGA)": np.mean(c_ab_vals) if c_ab_vals else None,
                    "C(NSGA, Alg)": np.mean(c_ba_vals) if c_ba_vals else None
                }
                summary_data.append(row)

    if summary_data:
        df = pd.DataFrame(summary_data)
        pd.options.display.float_format = '{:,.4f}'.format
        print("\n=== RESUMEN DE RESULTADOS ===")
        print(df.to_string(index=False))
        out_csv = config.DATA_DIR / "metrics_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[OK] Tabla guardada en: {out_csv}")
    else:
        print("\n[AVISO] No se encontraron datos.")

if __name__ == "__main__":
    main()