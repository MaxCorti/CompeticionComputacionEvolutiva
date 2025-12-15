# config.py
from pathlib import Path

# Directorios para "Tener organizados todos mis ficheros" (Punto 9)
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "experiments"
NSGA_DIR = BASE_DIR / "nsga2"
FIGURES_DIR = BASE_DIR / "figures"

# "He probado varias semillas para comprobar estabilidad" (Punto 3)
SEEDS = list(range(1, 11)) 
BUDGETS = [4000, 10000]

# Punto de referencia para Hipervolumen (Punto 6)
# Para ZDT3 (minimización), el punto ideal es (0,0) y el nadir real es aprox (0.85, 1.0).
# Usamos (1.1, 1.1) para asegurar que cubrimos todo el frente válido.
HV_REF_POINT = [1.1, 1.1]

# Experimentos
EXPERIMENTS = {
    "BASELINE": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 20.0, "pm": 1/30
    },
    "INFERIOR_SIG": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 10.0, "pm": 1/30
    },
    "SUPERIOR_SIG": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 30.0, "pm": 1/30
    },
    "INFERIOR_PM": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 20.0, "pm": 1/40
    },
    "SUPERIOR_PM": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 20.0, "pm": 1/20
    },
    "INFERIOR_CR": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.2, "SIG": 20.0, "pm": 1/30
    },
    "SUPERIOR_CR": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.8, "SIG": 20.0, "pm": 1/30
    },
    "INFERIOR_F": {
        "N": 100, "T": 20, "F": 0.2, "CR": 0.5, "SIG": 20.0, "pm": 1/30
    },
    "SUPERIOR_F": {
        "N": 100, "T": 20, "F": 0.8, "CR": 0.5, "SIG": 20.0, "pm": 1/30
    },
    "INFERIOR_T": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 20.0, "pm": 1/30
    },
    "SUPERIOR_T": {
        "N": 100, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 20.0, "pm": 1/30
    },
    "N40": {
        "N": 40, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 20.0, "pm": 1/30
    },
    "N80G50": {
        "N": 80, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 20.0, "pm": 1/30
    },
    "N200G50": {
        "N": 200, "T": 20, "F": 0.5, "CR": 0.5, "SIG": 20.0, "pm": 1/30
    }
}

NSGA_MAPPING = {4000: "P100G40", 10000: "P100G100"}