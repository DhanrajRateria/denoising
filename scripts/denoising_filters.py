import cv2
import numpy as np
import os
from isp_pipeline import load_bayer_raw, normalize_image, demosaic_bayer_image, apply_gamma_correction

# Define paths
INPUT_PATH = '../data/input/image.raw'
OUTPUT_PATH = '../output/'
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1280

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Denoising filter functions
def apply_median_filter(image):
    return cv2.medianBlur(image, 3)

def apply_bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Load and preprocess the Bayer RAW image using ISP pipeline
raw_image = load_bayer_raw(INPUT_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)
normalized_image = normalize_image(raw_image)
rgb_image = demosaic_bayer_image(normalized_image)
gamma_corrected_image = apply_gamma_correction(rgb_image)

# Apply denoising filters on the gamma-corrected RGB image
median_filtered = apply_median_filter(gamma_corrected_image)
bilateral_filtered = apply_bilateral_filter(gamma_corrected_image)
gaussian_filtered = apply_gaussian_filter(gamma_corrected_image)

# Save denoised images
cv2.imwrite(os.path.join(OUTPUT_PATH, 'median_filtered.png'), median_filtered)
cv2.imwrite(os.path.join(OUTPUT_PATH, 'bilateral_filtered.png'), bilateral_filtered)
cv2.imwrite(os.path.join(OUTPUT_PATH, 'gaussian_filtered.png'), gaussian_filtered)
print("Denoised images after ISP pipeline saved to output directory.")