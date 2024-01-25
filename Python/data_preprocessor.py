import os
import numpy as np
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

def normalize_matrix(matrix): # 0 ~ 1
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix
