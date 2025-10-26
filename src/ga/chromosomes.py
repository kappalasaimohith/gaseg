# src/ga/chromosomes.py
import numpy as np

def random_continuous_chrom(length):
    return np.random.rand(length)

def random_binary_chrom(length):
    return np.random.choice([0,1], size=(length,))
