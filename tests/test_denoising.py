import pytest
import numpy as np
from src.denoising.classical import ClassicalDenoising
from src.utils.image_utils import ImageUtils

@pytest.fixture
def test_image():
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)

def test_gaussian_filter():
    denoiser = ClassicalDenoising()
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    filtered = denoiser.gaussian_filter(image)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype

def test_median_filter():
    denoiser = ClassicalDenoising()
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    filtered = denoiser.median_filter(image)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype

def test_bilateral_filter():
    denoiser = ClassicalDenoising()
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    filtered = denoiser.bilateral_filter(image)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype

def test_snr_calculation():
    denoiser = ClassicalDenoising()
    clean = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    noisy = clean + np.random.normal(0, 10, clean.shape).astype(np.uint8)
    snr = denoiser.calculate_snr(clean, noisy)
    assert isinstance(snr, float)
    assert snr > 0