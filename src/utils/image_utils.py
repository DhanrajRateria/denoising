import cv2
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def load_image(path: str, grayscale: bool = True) -> np.ndarray:
        """
        Load image from path.
        
        Args:
            path: Path to image file
            grayscale: Whether to load as grayscale
            
        Returns:
            Loaded image array
        """
        if grayscale:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.imread(path)
    
    @staticmethod
    def save_image(image: np.ndarray, path: str) -> None:
        """
        Save image to path.
        
        Args:
            image: Image array to save
            path: Path to save image to
        """
        cv2.imwrite(path, image)
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_type: str = 'gaussian',
                  mean: float = 0, var: float = 0.01) -> np.ndarray:
        """
        Add noise to image.
        
        Args:
            image: Input image array
            noise_type: Type of noise ('gaussian' or 'salt_pepper')
            mean: Mean of Gaussian noise
            var: Variance of Gaussian noise
            
        Returns:
            Noisy image array
        """
        if noise_type == "gaussian":
            row, col = image.shape
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col))
            noisy = image + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        elif noise_type == "salt_pepper":
            row, col = image.shape
            s_vs_p = 0.5
            amount = 0.004
            noisy = np.copy(image)
            
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                     for i in image.shape]
            noisy[tuple(coords)] = 255
            
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                     for i in image.shape]
            noisy[tuple(coords)] = 0
            
            return noisy
        
        else:
            raise ValueError("Unsupported noise type")
    
    @staticmethod
    def plot_comparison(images: list, titles: list, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot multiple images for comparison.
        
        Args:
            images: List of image arrays
            titles: List of titles for each image
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()