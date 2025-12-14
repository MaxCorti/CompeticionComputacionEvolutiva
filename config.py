# config.py
from pathlib import Path

# --- Directorios ---
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "experiments"
NSGA_DIR = DATA_DIR / "nsga2_reference" 
FIGURES_DIR = BASE_DIR / "figures"

# --- Parámetros de Control Global ---
SEEDS = list(range(1, 11))  # Ejecutar semillas 1 a 10
BUDGETS = [4000, 10000]     # Límites de evaluaciones (Presupuestos)

# --- Definición de Experimentos ---
# Modifica estos diccionarios para probar distintas configuraciones.
# El algoritmo calculará automáticamente las Generaciones (G) = Budget / N
EXPERIMENTS = {
    # Configuración Base (Recomendada: P=100)
    # Para Budget 4000 -> G=40
    # Para Budget 10000 -> G=100
    "BASELINE": {
        "N": 100,           # Tamaño Población (Subproblemas)
        "T": 20,            # Tamaño Vecindad (20% de N)
        "F": 0.5,           # Factor mutación DE
        "CR": 0.5,          # Prob. cruce DE
        "SIG": 20.0,        # Sigma para mutación Gaussiana
        "pm": 1/30,         # Prob. mutación por variable (1/dim)
    },
    
    # Variante con Población Reducida (P=80)
    # Para Budget 4000 -> G=50
    "P80_CONFIG": {
        "N": 80,            # Al bajar N, aumentan las generaciones disponibles
        "T": 16,            # Ajustamos vecindad al 20% de 80
        "F": 0.5,
        "CR": 0.5,
        "SIG": 20.0,
        "pm": 1/30,
    },

    # Variante Alta Exploración
    "HIGH_MUTATION": {
        "N": 100, "T": 20, "F": 0.9, "CR": 0.9, "SIG": 10.0, "pm": 1/30
    },
}

# Mapeo para localizar los archivos de NSGA-II según el presupuesto
NSGA_MAPPING = {
    4000: "P100G40",   # Nombre de la carpeta de NSGA-II para 4000 evals
    10000: "P100G100"  # Nombre de la carpeta de NSGA-II para 10000 evals
}