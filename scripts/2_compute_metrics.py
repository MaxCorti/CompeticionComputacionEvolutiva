# scripts/2_compute_metrics.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from pathlib import Path
import config

# --- UTILIDADES ---
def load_front(path):
    if not path.exists(): return None
    # Cargar y extraer SOLO las ultimas 2 columnas (f1, f2)
    return np.loadtxt(path)[:, -2:] 

def get_nondominated(front):
    """Filtra soluciones no dominadas de un frente."""
    costs = front
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | \
                                         np.any(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True
    return costs[is_efficient]

# --- MÉTRICAS ---
def spacing(front):
    """Métrica Spacing (uniformidad). Menor es mejor."""
    if len(front) < 2: return 0.0
    # Ordenar por f1
    front = front[np.argsort(front[:, 0])]
    # Distancias Manhattan entre puntos consecutivos (aprox para 2D)
    dists = np.sum(np.abs(front[:-1] - front[1:]), axis=1)
    d_mean = np.mean(dists)
    return np.sqrt(np.mean((dists - d_mean)**2))

def hypervolume_indicator(front, ref_point=np.array([1.1, 1.1])):
    """Cálculo simple de Hipervolumen 2D."""
    # Filtrar dominados y ordenar
    front = get_nondominated(front)
    front = front[np.argsort(front[:, 0])]
    
    # Filtrar puntos que exceden el punto de referencia
    front = front[(front[:,0] <= ref_point[0]) & (front[:,1] <= ref_point[1])]
    
    if len(front) == 0: return 0.0
    
    hv = 0.0
    # Barrido
    max_y = ref_point[1]
    for p in front:
        width = ref_point[0] - p[0]
        height = max_y - p[1]
        if height > 0:
            hv += width * height
            max_y = p[1] # Actualizar techo para el siguiente punto
    return hv

def main():
    summary_data = []
    
    print("--- CALCULANDO MÉTRICAS ---")
    
    for exp_name in config.EXPERIMENTS.keys():
        for budget in config.BUDGETS:
            hv_list = []
            sp_list = []
            
            for seed in config.SEEDS:
                # Ruta fichero MOEA/D
                f_path = config.RESULTS_DIR / exp_name / str(budget) / f"seed_{seed:02d}.txt"
                front = load_front(f_path)
                
                if front is None: continue
                
                # Calcular métricas
                hv = hypervolume_indicator(front)
                sp = spacing(front)
                
                hv_list.append(hv)
                sp_list.append(sp)
            
            if hv_list:
                summary_data.append({
                    "Experimento": exp_name,
                    "Presupuesto": budget,
                    "HV (Media)": np.mean(hv_list),
                    "HV (Std)": np.std(hv_list),
                    "Spacing (Media)": np.mean(sp_list),
                    "Spacing (Std)": np.std(sp_list)
                })
                
    # Crear DataFrame y guardar
    df = pd.DataFrame(summary_data)
    print("\nRESUMEN DE RESULTADOS:")
    print(df.to_string(index=False))
    
    out_csv = config.DATA_DIR / "metrics_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nTabla guardada en: {out_csv}")

if __name__ == "__main__":
    main()