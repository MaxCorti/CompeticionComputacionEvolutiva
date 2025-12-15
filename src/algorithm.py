# src/algorithm.py
import numpy as np

from src.decomposition import (compute_neighbors, generate_weight_vectors,
                               tchebycheff)
from src.operators import differential_evolution_op, gaussian_mutation
from src.problem import ZDT3


class MOEAD:
    def __init__(self, params, seed):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.problem = ZDT3()
        self.N = params['N']
        self.T = params['T']
        
        self.weights = generate_weight_vectors(self.N)
        self.neighbors = compute_neighbors(self.weights, self.T)
        self.population = np.array([self.problem.random_solution(self.rng) for _ in range(self.N)])
        self.objectives = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        # "Tengo claro qué es el punto z* y cómo se actualiza" (Punto 1)
        self.z_star = np.min(self.objectives, axis=0)
        self.evaluations = self.N

    def run(self, max_evals):
        while self.evaluations < max_evals:
            for i in range(self.N):
                if self.evaluations >= max_evals: break
                
                # Reproducción + Mutación
                offspring = differential_evolution_op(i, self.population, self.neighbors[i], 
                                                    self.params['F'], self.params['CR'], self.rng)
                offspring = gaussian_mutation(offspring, self.problem.lower, self.problem.upper,
                                            self.params['pm'], self.params['SIG'], self.rng)
                
                f_offspring = self.problem.evaluate(offspring)
                self.evaluations += 1
                
                # Actualización de z*
                self.z_star = np.min(np.vstack((self.z_star, f_offspring)), axis=0)
                
                # Actualización de Vecinos
                for j in self.neighbors[i]:
                    g_old = tchebycheff(self.objectives[j], self.weights[j], self.z_star)
                    g_new = tchebycheff(f_offspring,       self.weights[j], self.z_star)
                    if g_new <= g_old:
                        self.population[j] = offspring
                        self.objectives[j] = f_offspring
                        
        return self.population, self.objectives