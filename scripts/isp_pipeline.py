import cv2
import numpy as np
import os

# Define paths and constants
INPUT_PATH = '../data/input/image.raw'
OUTPUT_PATH = '../output/'
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1280
BAYER_PATTERN = cv2.COLOR_BAYER_GR2BGR  # GRBG Bayer pattern

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_bayer_raw(path, width, height):
    """Load 12-bit Bayer RAW image and reshape it to 2D."""
    raw_image = np.fromfile(path, dtype=np.uint16)
    raw_image = raw_image.reshape((height, width))
    return raw_image

def normalize_image(image, max_value=4095):
    """Normalize a 12-bit image to 8-bit."""
    return cv2.convertScaleAbs(image, alpha=(255.0 / max_value))

def demosaic_bayer_image(bayer_image):
    """Convert Bayer image to RGB using demosaicing."""
    return cv2.cvtColor(bayer_image, BAYER_PATTERN)

def apply_gamma_correction(image, gamma=2.2):
    """Apply gamma correction to enhance contrast."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Load and preprocess the Bayer RAW image
raw_image = load_bayer_raw(INPUT_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)

# Normalize to 8-bit for viewing and processing
normalized_image = normalize_image(raw_image)

# Demosaic the image to convert from Bayer to RGB
rgb_image = demosaic_bayer_image(normalized_image)

# Apply gamma correction
gamma_corrected_image = apply_gamma_correction(rgb_image)

# Save the RGB image
cv2.imwrite(os.path.join(OUTPUT_PATH, 'rgb_image.png'), gamma_corrected_image)
print("ISP processed RGB image saved to output directory.")