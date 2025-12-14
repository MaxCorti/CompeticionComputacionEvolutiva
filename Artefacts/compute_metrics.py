# compute_metrics.py
import numpy as np
from pathlib import Path
from math import sqrt

# ---------------- CONFIGURACIÓN ----------------

SEEDS = list(range(1, 11))  # 10 ejecuciones

# Directorios base
AGG_BASE = Path("results") / "ZDT3"
NSGA_BASE = Path("nsga2")

# Agregación: una configuración por presupuesto
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

# Configs NSGA-II disponibles en EV para cada presupuesto
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

# Punto de referencia para hipervolumen (ajústalo si quieres)
HV_REF = np.array([1.2, 1.2])


# ---------------- UTILIDADES BÁSICAS ----------------

def load_front_agg(path: Path) -> np.ndarray:
    """
    Carga frente de tu algoritmo (final_pop_seedXX.txt)
    Asume dos últimas columnas = f1, f2.
    """
    data = np.loadtxt(path)
    return data[:, -2:]


def load_front_nsga(dir_path: Path, N: int, G: int, seed: int) -> np.ndarray:
    """
    Carga frente de NSGA-II desde archivos tipo:
    zdt3_final_popp40g250_seed01.out
    donde tag = p{N}g{G} en minúsculas.
    """
    tag = f"p{N}g{G}".lower()
    filename = f"zdt3_final_pop{tag}_seed{seed:02d}.out"
    path = dir_path / filename
    data = np.loadtxt(path)
    # Normalmente ya vienen f1, f2, pero nos quedamos con las dos últimas columnas por si acaso
    return data[:, -2:]


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True si a domina a b (minimización)."""
    return np.all(a <= b) and np.any(a < b)


def nondominated_front(F: np.ndarray) -> np.ndarray:
    """Devuelve solo los puntos no dominados de F."""
    n = F.shape[0]
    is_dom = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dom[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if dominates(F[j], F[i]):
                is_dom[i] = True
                break
    return F[~is_dom]


# ---------------- MÉTRICAS ----------------

def spacing(F: np.ndarray) -> float:
    """
    Spacing clásico: mide uniformidad de las distancias al vecino más cercano.
    """
    N = F.shape[0]
    if N <= 1:
        return 0.0

    d = np.full(N, np.inf)
    for i in range(N):
        diff = F - F[i]
        dist = np.sqrt((diff ** 2).sum(axis=1))
        dist[i] = np.inf
        d[i] = dist.min()
    d_mean = d.mean()
    return float(sqrt(((d - d_mean) ** 2).sum() / (N - 1)))


def hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float:
    """
    Hipervolumen 2D para minimización.
    F: frente no dominado (o se filtrará).
    ref: punto de referencia [r1, r2].
    """
    if F.size == 0:
        return 0.0

    mask = (F[:, 0] <= ref[0]) & (F[:, 1] <= ref[1])
    P = F[mask]
    if P.size == 0:
        return 0.0

    P = nondominated_front(P)
    P = P[np.argsort(P[:, 0])]  # orden por f1 asc.

    hv = 0.0
    min_f2 = ref[1]
    # recorre de derecha a izquierda
    for f1, f2 in P[::-1]:
        width = ref[0] - f1
        height = max(0.0, min_f2 - f2)
        if width > 0 and height > 0:
            hv += width * height
        if f2 < min_f2:
            min_f2 = f2
    return float(hv)


def coverage(A: np.ndarray, B: np.ndarray) -> float:
    """
    Coverage C(A,B) = fracción de puntos de B dominados por al menos un punto de A.
    """
    if B.size == 0:
        return 0.0
    count = 0
    for b in B:
        if any(dominates(a, b) for a in A):
            count += 1
    return count / B.shape[0]


def mean_std(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ---------------- MAIN ----------------

def main():
    for budget in [4000, 10000]:
        print("\n" + "=" * 80)
        print(f"PRESUPUESTO: {budget} evaluaciones")
        print("=" * 80)

        agg_cfg = AGG_CONFIGS[budget]
        agg_dir = agg_cfg["dir"]

        # --- Cargamos todos los frentes de agregación (una vez) ---
        agg_fronts = []
        for seed in SEEDS:
            agg_file = agg_dir / f"final_pop_seed{seed:02d}.txt"
            if not agg_file.exists():
                print(f"[AVISO] No existe {agg_file}, se omite seed {seed}.")
                continue
            F_agg = load_front_agg(agg_file)
            agg_fronts.append(F_agg)

        if not agg_fronts:
            print("[ERROR] No hay frentes de agregación para este presupuesto.")
            continue

        # Métricas de agregación sola (por si quieres ponerlas en tabla)
        agg_spacing_vals = [spacing(F) for F in agg_fronts]
        agg_hv_vals = [hypervolume_2d(nondominated_front(F), HV_REF) for F in agg_fronts]
        m_s_a, sd_s_a = mean_std(agg_spacing_vals)
        m_h_a, sd_h_a = mean_std(agg_hv_vals)

        print("\n>>> Métricas SOLO Agregación (comparativas internas)")
        print(f"Spacing   : media = {m_s_a:.4e}, std = {sd_s_a:.4e}")
        print(f"HiperVol. : media = {m_h_a:.4e}, std = {sd_h_a:.4e}")

        # --- Para cada configuración de NSGA-II de este presupuesto ---
        for nsga_cfg in NSGA_CONFIGS[budget]:
            name = nsga_cfg["name"]
            N = nsga_cfg["N"]
            G = nsga_cfg["G"]

            print("\n" + "-" * 60)
            print(f"NSGA-II: {name} (N={N}, G={G}, N*G={N*G})")
            print("-" * 60)

            nsga_dir = NSGA_BASE / f"ZDT3_{budget}" / name

            nsga_spacing_vals = []
            nsga_hv_vals = []
            cov_agg_nsga_vals = []
            cov_nsga_agg_vals = []

            for seed in SEEDS:
                try:
                    # Cargamos frente de NSGA-II
                    F_nsga = load_front_nsga(nsga_dir, N, G, seed)
                except OSError:
                    print(f"[AVISO] No se ha encontrado fichero NSGA para seed {seed} en {nsga_dir}")
                    continue

                # Frente de agregación correspondiente a esa seed
                # (suponemos mismo orden de seeds)
                agg_file = agg_dir / f"final_pop_seed{seed:02d}.txt"
                if not agg_file.exists():
                    print(f"[AVISO] No hay frente de agregación para seed {seed}, se omite esta pareja.")
                    continue
                F_agg = load_front_agg(agg_file)

                # Spacing
                nsga_spacing_vals.append(spacing(F_nsga))

                # Hipervolumen
                nsga_hv_vals.append(hypervolume_2d(nondominated_front(F_nsga), HV_REF))

                # Coverage
                cov_agg_nsga_vals.append(coverage(F_agg, F_nsga))
                cov_nsga_agg_vals.append(coverage(F_nsga, F_agg))

            # Estadísticos
            m_s_n, sd_s_n = mean_std(nsga_spacing_vals)
            m_h_n, sd_h_n = mean_std(nsga_hv_vals)
            m_c_an, sd_c_an = mean_std(cov_agg_nsga_vals)
            m_c_na, sd_c_na = mean_std(cov_nsga_agg_vals)

            print("Spacing:")
            print(f"  Agregación (ref)  : media = {m_s_a:.4e}, std = {sd_s_a:.4e}")
            print(f"  NSGA-II {name}    : media = {m_s_n:.4e}, std = {sd_s_n:.4e}")

            print("\nHipervolumen (ref = "
                  f"[{HV_REF[0]:.2f}, {HV_REF[1]:.2f}]):")
            print(f"  Agregación (ref)  : media = {m_h_a:.4e}, std = {sd_h_a:.4e}")
            print(f"  NSGA-II {name}    : media = {m_h_n:.4e}, std = {sd_h_n:.4e}")

            print("\nCoverage:")
            print("  C(Agregación, NSGA-II)  = fracción de puntos de NSGA dominados por Agregación")
            print(f"    media = {m_c_an:.4f}, std = {sd_c_an:.4f}")
            print("  C(NSGA-II, Agregación)  = fracción de puntos de Agregación dominados por NSGA")
            print(f"    media = {m_c_na:.4f}, std = {sd_c_na:.4f}")

    print("\nFin del cálculo de métricas.")


if __name__ == "__main__":
    main()
