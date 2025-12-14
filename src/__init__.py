# src/__init__.py
from .algorithm import MOEAD
from .decomposition import (compute_neighbors, generate_weight_vectors,
                            tchebycheff)
from .operators import differential_evolution_op, gaussian_mutation
from .problem import ZDT3
