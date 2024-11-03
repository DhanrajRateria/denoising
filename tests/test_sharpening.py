import pytest
import numpy as np
from src.sharpening.edge_enhancement import ImageSharpening

@pytest.fixture
def test_image():
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)

def test_unsharp_mask():
    sharpener = ImageSharpening()
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    sharpened = sharpener.unsharp_mask(image)
    assert sharpened.shape == image.shape
    assert sharpened.dtype == image.dtype

def test_laplacian_sharpening():
    sharpener = ImageSharpening()
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    sharpened = sharpener.laplacian_sharpening(image)
    assert sharpened.shape == image.shape
    assert sharpened.dtype == image.dtype

def test_edge_strength():
    sharpener = ImageSharpening()
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    strength = sharpener.calculate_edge_strength(image)
    assert isinstance(strength, float)
    assert strength >= 0