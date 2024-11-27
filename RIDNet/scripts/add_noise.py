import random
import numpy as np
import os
from glob import glob
import cv2

"""

The dataset is clean so artificial noise will be added to simulate real-world scenarios


"""


# Function to add different types of noise to an image
def add_noise(image, noise_type="gaussian"):
    if noise_type == "gaussian":
        # Gaussian noise: Mean-centered random values with a standard deviation
        mean = 0
        std = 0.1  # Standard deviation
        gaussian = np.random.normal(mean, std, image.shape)  # Generate Gaussian noise
        noisy_image = np.clip(image + gaussian, 0, 1)  # Add noise and clip values to [0, 1]

    elif noise_type == "poisson":
        # Poisson noise: Based on random sampling from a Poisson distribution
        vals = len(np.unique(image))  # Determine unique pixel values in the image
        vals = 2 ** np.ceil(np.log2(vals))  # Ensure the number of values is a power of 2
        noisy_image = np.clip(np.random.poisson(image * vals) / vals, 0, 1)  # Add noise and normalize

    elif noise_type == "speckle":
        # Speckle noise: Multiplicative noise proportional to the image's intensity
        noise = np.random.randn(*image.shape)  # Generate random values (normal distribution)
        noisy_image = np.clip(image + image * noise, 0, 1)  # Add noise and clip values to [0, 1]

    else:
        # Handle unsupported noise types
        raise ValueError("Unsupported noise type")

    return noisy_image


def apply_noise(input_folder, output_folder, noise_type="gaussian"):
    pass
