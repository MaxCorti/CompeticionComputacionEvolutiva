from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIGURACIÓN ----------------

SEED_TO_PLOT = 1

AGG_BASE = Path("results") / "ZDT3"
NSGA_BASE = Path("nsga2")

AGG_CONFIGS = {
    4000:  {"name": "Agregacion_P100G40",  "dir": AGG_BASE / "P100G40"},
    10000: {"name": "Agregacion_P100G100", "dir": AGG_BASE / "P100G100"},
}

NSGA_CONFIGS = {
    4000: [
        {"name": "P40G100", "N": 40, "G": 100},
        {"name": "P80G50",  "N": 80, "G": 50},
        {"name": "P100G40", "N": 100,"G": 40},
    ],
    10000: [
        {"name": "P40G250", "N": 40,  "G": 250},
        {"name": "P100G100","N": 100, "G": 100},
        {"name": "P200G50", "N": 200, "G": 50},
    ],
}

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ---------------- UTILIDADES ----------------

def load_front_agg(agg_dir: Path, seed: int) -> np.ndarray:
    fname = agg_dir / f"final_pop_seed{seed:02d}.txt"
    if not fname.exists():
        raise FileNotFoundError(f"Falta fichero: {fname}")
    # Formato Agregación: Vars(30) + Objs(2). Las últimas 2 son los objetivos.
    return np.loadtxt(fname)[:, -2:]


def load_front_nsga(budget: int, cfg: dict, seed: int) -> np.ndarray:
    N, G = cfg["N"], cfg["G"]
    tag = f"p{N}g{G}".lower()
    dir_path = NSGA_BASE / f"ZDT3_{budget}" / cfg["name"]
    fname = dir_path / f"zdt3_final_pop{tag}_seed{seed:02d}.out"
    if not fname.exists():
        raise FileNotFoundError(f"Falta fichero NSGA: {fname}")
    
    data = np.loadtxt(fname)

    return data[:, 0:2]


def generate_zdt3_true_front(num_points: int = 5000) -> np.ndarray:
    """Genera frente real discontinuo ZDT3 filtrando dominados visualmente."""
    x = np.linspace(0.0, 1.0, num_points)
    f1 = x
    f2 = 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)
    points = np.vstack([f1, f2]).T
    
    # Filtro simple para limpiar las partes dominadas de la onda
    clean_points = []
    points = points[np.argsort(points[:, 0])] # Ordenar por f1
    
    for i, p in enumerate(points):
        # Si un punto tiene un vecino a la izquierda con f2 mucho menor, es dominado (parte alta de la onda)
        # Una heurística rápida para ZDT3 es verificar si hay puntos con f1 menor y f2 menor.
        if np.any((points[:,0] <= p[0]) & (points[:,1] < p[1])):
             continue
        clean_points.append(p)
    return np.array(clean_points)


def plot_single_config(budget: int, nsga_cfg: dict, seed: int):
    agg_cfg = AGG_CONFIGS[budget]
    
    try:
        F_agg = load_front_agg(agg_cfg["dir"], seed)
        F_nsga = load_front_nsga(budget, nsga_cfg, seed)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    F_true = generate_zdt3_true_front()

    plt.figure(figsize=(8, 6))

    # 1. Frente Real (Puntos rojos pequeños)
    plt.scatter(F_true[:, 0], F_true[:, 1], s=2, c='red', alpha=0.5, label="Frente real ZDT3")

    # 2. Agregación (Puntos azules)
    plt.scatter(F_agg[:, 0], F_agg[:, 1], s=25, c='dodgerblue', marker='o', 
                edgecolors='k', linewidth=0.5, alpha=0.9, label="Agregación")

    # 3. NSGA-II (Cruces naranjas)
    plt.scatter(F_nsga[:, 0], F_nsga[:, 1], s=30, c='darkorange', marker='x', 
                linewidth=1.5, alpha=0.9, label=f"NSGA-II ({nsga_cfg['name']})")

    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")
    plt.title(f"ZDT3 (Presupuesto: {budget}) - Seed {seed}\n{agg_cfg['name']} vs {nsga_cfg['name']}")
    
    # Escala ajustada a la región de interés del Frente ZDT3
    plt.xlim(0.0, 1.1)
    plt.ylim(-0.8, 1.2) # Ajustado para ZDT3 (f2 baja hasta -0.7 aprox)
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()

    out_name = f"front_budget{budget}_{agg_cfg['name']}_vs_{nsga_cfg['name']}_seed{seed:02d}.png"
    out_path = FIG_DIR / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generado: {out_name}")


def plot_all():
    for budget in [4000, 10000]:
        print(f"\n--- Presupuesto {budget} ---")
        for nsga_cfg in NSGA_CONFIGS[budget]:
            plot_single_config(budget, nsga_cfg, SEED_TO_PLOT)


if __name__ == "__main__":
    plot_all()