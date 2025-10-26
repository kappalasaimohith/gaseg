import numpy as np
from .operators import tournament_select, blend_crossover, gaussian_mutation
from src.eval.metrics import iou

class ThresholdGA:
    """
    Simple GA that evolves a single continuous threshold in [0,1] to binarize
    anomaly maps and maximize mean IoU on a validation set.
    This is intentionally minimal and intended as scaffolding for the full GA-SEG.
    """
    def __init__(self, eval_fn, pop_size=20, generations=20, elitism=0.1, seed=None):
        """
        eval_fn: callable(threshold) -> list of IoU scores (one per image) or mean IoU
        """
        if seed is not None:
            np.random.seed(seed)
        self.eval_fn = eval_fn
        self.pop_size = pop_size
        self.generations = generations
        self.elitism = max(1, int(pop_size * elitism))
        # population of scalars in [0,1]
        self.population = np.random.rand(pop_size)

    def run(self, verbose=True):
        best = None
        history = []
        for gen in range(self.generations):
            fitness = []
            for indiv in self.population:
                # eval_fn should return mean IoU or list
                val = self.eval_fn(indiv)
                if isinstance(val, (list, tuple, np.ndarray)):
                    score = float(np.nanmean(val))
                else:
                    score = float(val)
                fitness.append(score)

            idx_sorted = np.argsort(fitness)[::-1]
            gen_best = fitness[idx_sorted[0]]
            history.append(gen_best)
            if best is None or gen_best > best[0]:
                best = (gen_best, float(self.population[idx_sorted[0]]), fitness)
            if verbose:
                print(f"GA Gen {gen+1}/{self.generations} - best IoU: {gen_best:.4f} - mean IoU: {np.mean(fitness):.4f}")

            # elitism keep
            new_pop = list(self.population[idx_sorted[:self.elitism]])
            # fill rest
            while len(new_pop) < self.pop_size:
                a = tournament_select(fitness, k=3)
                b = tournament_select(fitness, k=3)
                pa = self.population[a]
                pb = self.population[b]
                child = blend_crossover(np.array([pa]), np.array([pb]), alpha=0.2)[0]
                child = gaussian_mutation(child.reshape(1,), sigma=0.02, p_mut=0.2)[0]
                child = np.clip(child, 0.0, 1.0)
                new_pop.append(child)
            self.population = np.array(new_pop)

        return best, history
