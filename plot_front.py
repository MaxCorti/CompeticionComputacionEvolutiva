# plot_front.py
import numpy as np
import matplotlib.pyplot as plt

def load_population(filename):
    """
    Carga el fichero final_pop...txt generado por el algoritmo.
    Últimas columnas = f1, f2.
    """
    data = np.loadtxt(filename)
    f1 = data[:, -2]
    f2 = data[:, -1]
    return f1, f2

def plot_front(f1, f2, title="Frente obtenido (Algoritmo de Agregación)"):
    plt.figure(figsize=(6, 5))
    plt.scatter(f1, f2, s=12)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def zdt3_true_front(n_points=500):
    """Genera una aproximación al frente verdadero de ZDT3 (analítico)."""
    import numpy as np

    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
    return f1, f2


if __name__ == "__main__":
    filename = "final_pop_ZDT3_N100_G100.txt"
    f1, f2 = load_population(filename)
    f1_true, f2_true = zdt3_true_front()

    plt.figure(figsize=(6, 5))
    plt.scatter(f1, f2, label="Algoritmo Agregación", s=12)
    plt.plot(f1_true, f2_true, "r--", label="Frente real ZDT3", linewidth=1.5)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Comparación frente obtenido vs frente real")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

