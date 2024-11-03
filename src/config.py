import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model parameters
UNET_INPUT_SIZE = (256, 256)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Processing parameters
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0
MEDIAN_KERNEL_SIZE = 5
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75.0
BILATERAL_SIGMA_SPACE = 75.0
UNSHARP_AMOUNT = 1.5
LAPLACIAN_KERNEL_SIZE = 3

# Noise parameters
NOISE_MEAN = 0
NOISE_VAR = 0.01