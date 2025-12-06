# plot_fronts_compare.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIGURACIÓN ----------------

# Seed que quieres usar para las gráficas (puedes cambiarla)
SEED_TO_PLOT = 1

# Directorios base (coinciden con lo que hemos usado en los otros scripts)
AGG_BASE = Path("results") / "ZDT3"
NSGA_BASE = Path("nsga2")

# Agregación: una config por presupuesto
AGG_CONFIGS = {
    4000: {
        "name": "Agregacion_P100G40",
        "dir": AGG_BASE / "P100G40",
    },
    10000: {
        "name": "Agregacion_P100G100",
        "dir": AGG_BASE / "P100G100",
    },
}

# Configs NSGA-II (como las tienes en EV)
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

# Carpeta donde se guardarán las figuras
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ---------------- UTILIDADES ----------------

def load_front_agg(agg_dir: Path, seed: int) -> np.ndarray:
    """
    Carga el frente de tu algoritmo (final_pop_seedXX.txt).
    Devuelve un array (N, 2) con [f1, f2].
    """
    fname = agg_dir / f"final_pop_seed{seed:02d}.txt"
    data = np.loadtxt(fname)
    return data[:, -2:]


def load_front_nsga(budget: int, cfg: dict, seed: int) -> np.ndarray:
    """
    Carga frente de NSGA-II desde archivos tipo:
    nsga2/ZDT3_4000/P40G100/zdt3_final_popp40g100_seed01.out
    """
    N = cfg["N"]
    G = cfg["G"]
    tag = f"p{N}g{G}".lower()

    dir_path = NSGA_BASE / f"ZDT3_{budget}" / cfg["name"]
    fname = dir_path / f"zdt3_final_pop{tag}_seed{seed:02d}.out"
    data = np.loadtxt(fname)
    return data[:, -2:]


def generate_zdt3_true_front(num_points: int = 1000) -> np.ndarray:
    """
    Genera una aproximación del frente real de ZDT3 (g=1).
    f1 = x
    f2 = 1 - sqrt(x) - x * sin(10*pi*x)
    El frente es discontinuo, pero para visualizar basta con muestrear denso.
    """
    x = np.linspace(0.0, 1.0, num_points)
    f1 = x
    f2 = 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)
    return np.vstack([f1, f2]).T


def plot_single_config(budget: int, nsga_cfg: dict, seed: int):
    """
    Genera una figura con:
    - Frente real de ZDT3
    - Agregación (para el presupuesto correspondiente)
    - NSGA-II (config específica)
    """
    agg_cfg = AGG_CONFIGS[budget]
    agg_dir = agg_cfg["dir"]

    # Cargar frentes
    F_true = generate_zdt3_true_front()
    F_agg = load_front_agg(agg_dir, seed)
    F_nsga = load_front_nsga(budget, nsga_cfg, seed)

    # Figura
    plt.figure(figsize=(7, 5))

    # Frente real
    plt.plot(F_true[:, 0], F_true[:, 1],
             linestyle='--', linewidth=1.5, label="Frente real ZDT3")

    # Agregación
    plt.scatter(F_agg[:, 0], F_agg[:, 1],
                s=15, marker='o', alpha=0.7, label=f"Agregación ({agg_cfg['name']}, seed {seed:02d})")

    # NSGA-II
    plt.scatter(F_nsga[:, 0], F_nsga[:, 1],
                s=15, marker='x', alpha=0.7, label=f"NSGA-II {nsga_cfg['name']} (seed {seed:02d})")

    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(f"Comparación de frentes - Presupuesto {budget} evals\nAgregación vs {nsga_cfg['name']} (NSGA-II)")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend()

    # Guardar figura
    out_name = f"front_budget{budget}_{agg_cfg['name']}_vs_{nsga_cfg['name']}_seed{seed:02d}.png"
    out_path = FIG_DIR / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Guardada figura: {out_path}")


def plot_all():
    for budget in [4000, 10000]:
        print("\n" + "=" * 70)
        print(f"Generando figuras para presupuesto {budget} evaluaciones (seed={SEED_TO_PLOT})")
        print("=" * 70)

        for nsga_cfg in NSGA_CONFIGS[budget]:
            try:
                plot_single_config(budget, nsga_cfg, SEED_TO_PLOT)
            except OSError as e:
                print(f"[AVISO] No se ha podido generar figura para {nsga_cfg['name']}: {e}")


if __name__ == "__main__":
    plot_all()
