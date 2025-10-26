# src/ga/operators.py
import numpy as np

def tournament_select(fitness, k=3):
    import random
    participants = random.sample(range(len(fitness)), k)
    best = max(participants, key=lambda i: fitness[i])
    return best

def single_point_crossover(a, b):
    # a, b numpy arrays
    L = len(a)
    if L == 1:
        return a.copy()
    pt = np.random.randint(1, L)
    child = np.concatenate([a[:pt], b[pt:]])
    return child

def blend_crossover(a, b, alpha=0.5):
    # for continuous genes
    gamma = np.random.uniform(-alpha, 1+alpha, size=a.shape)
    child = (1-gamma)*a + gamma*b
    return child

def gaussian_mutation(chrom, mu=0.0, sigma=0.05, p_mut=0.1):
    child = chrom.copy()
    mask = np.random.rand(*chrom.shape) < p_mut
    noise = np.random.normal(mu, sigma, size=chrom.shape)
    child[mask] = child[mask] + noise[mask]
    child = np.clip(child, 0.0, 1.0)
    return child
