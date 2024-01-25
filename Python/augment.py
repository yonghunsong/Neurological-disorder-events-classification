import os
import numpy as np
import random

# flipping
def flip(data):
    return np.flip(data)

# shifting
def shift(data, n):
    return np.roll(data, n)

# scaling
def scale(data, factor):
    return data * factor

# noise injection
def add_noise(data, noise_level):
    noise = np.random.randn(len(data)) * noise_level
    return data + noise

# time series data augmentation
def augment_data(data, n_augment=1, flip_prob=0.5, shift_max=10, scale_min=0.9, scale_max=1.1, noise_level=0.1):
    augmented_data = []
    for i in range(n_augment):
        # flipping
        if random.random() < flip_prob:
            augmented_data.append(flip(data))
        # shifting
        n_shift = random.randint(-shift_max, shift_max)
        augmented_data.append(shift(data, n_shift))
        # scaling
        scale_factor = random.uniform(scale_min, scale_max)
        augmented_data.append(scale(data, scale_factor))
        # noise injection
        augmented_data.append(add_noise(data, noise_level))
    return augmented_data
