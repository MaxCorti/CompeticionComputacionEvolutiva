# src/algorithm.py
import numpy as np
from src.problem import ZDT3
from src.decomposition import generate_weight_vectors, compute_neighbors, tchebycheff
from src.operators import differential_evolution_op, gaussian_mutation

class MOEAD:
    def __init__(self, params, seed):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.problem = ZDT3(dim=30)
        
        self.N = params['N']
        self.T = params['T']
        
        # Inicialización
        self.weights = generate_weight_vectors(self.N)
        self.neighbors = compute_neighbors(self.weights, self.T)
        
        # Población inicial
        self.population = np.array([self.problem.random_solution(self.rng) for _ in range(self.N)])
        self.objectives = np.array([self.problem.evaluate(ind) for ind in self.population])
        
        # Punto de referencia ideal z* (inicial)
        self.z_star = np.min(self.objectives, axis=0)
        
        # Contador de evaluaciones
        self.evaluations = self.N

    def run(self, max_evals):
        """Ejecuta el algoritmo hasta alcanzar max_evals."""
        
        while self.evaluations < max_evals:
            # Iterar por cada subproblema (individuo)
            for i in range(self.N):
                if self.evaluations >= max_evals:
                    break
                
                # 1. Reproducción (DE) usando vecindad
                offspring = differential_evolution_op(
                    i, self.population, self.neighbors[i], 
                    self.params['F'], self.params['CR'], self.rng
                )
                
                # 2. Mutación (Gaussiana)
                offspring = gaussian_mutation(
                    offspring, self.problem.lower, self.problem.upper,
                    self.params['pm'], self.params['SIG'], self.rng
                )
                
                # 3. Evaluación
                f_offspring = self.problem.evaluate(offspring)
                self.evaluations += 1
                
                # 4. Actualizar z*
                self.z_star = np.min(np.vstack((self.z_star, f_offspring)), axis=0)
                
                # 5. Actualizar Vecinos (Reemplazo por Tchebycheff)
                # Solo actualizamos si mejora el subproblema vecino
                for j in self.neighbors[i]:
                    g_old = tchebycheff(self.objectives[j], self.weights[j], self.z_star)
                    g_new = tchebycheff(f_offspring,       self.weights[j], self.z_star)
                    
                    if g_new <= g_old:
                        self.population[j] = offspring
                        self.objectives[j] = f_offspring
                        
        return self.population, self.objectives