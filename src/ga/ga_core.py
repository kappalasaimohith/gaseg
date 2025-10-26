# src/ga/ga_core.py
import numpy as np
from .operators import tournament_select, blend_crossover, gaussian_mutation

class GA:
    """
    Small GA engine for continuous chromosomes [0,1].
    fitness_fn receives chromosome and returns scalar score (higher better).
    """
    def __init__(self, fitness_fn, chrom_len, pop_size=40, generations=50,
                 elitism=0.05, device=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.fitness_fn = fitness_fn
        self.chrom_len = chrom_len
        self.pop_size = pop_size
        self.generations = generations
        self.elitism = max(1, int(pop_size * elitism))
        self.population = [np.random.rand(chrom_len) for _ in range(pop_size)]

    def run(self, verbose=True):
        best = None
        history = []
        for gen in range(self.generations):
            fitness = [self.fitness_fn(ch) for ch in self.population]
            idx_sorted = np.argsort(fitness)[::-1]
            best_idx = idx_sorted[0]
            gen_best_score = fitness[best_idx]
            gen_best_chrom = self.population[best_idx].copy()
            history.append(gen_best_score)
            if best is None or gen_best_score > best[0]:
                best = (gen_best_score, gen_best_chrom.copy(), fitness)
            if verbose:
                print(f"Gen {gen+1}/{self.generations} - best fitness: {gen_best_score:.4f} - mean: {np.mean(fitness):.4f}")
            # elitism
            new_pop = [self.population[i] for i in idx_sorted[:self.elitism]]
            # fill rest
            while len(new_pop) < self.pop_size:
                a = tournament_select(fitness, k=3)
                b = tournament_select(fitness, k=3)
                parent_a = self.population[a]
                parent_b = self.population[b]
                child = blend_crossover(parent_a, parent_b, alpha=0.2)
                child = gaussian_mutation(child, sigma=0.05, p_mut=0.1)
                new_pop.append(child)
            self.population = new_pop
        return best, history
