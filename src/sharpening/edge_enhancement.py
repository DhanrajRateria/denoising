import cv2
import numpy as np
from typing import Tuple, Union

class ImageSharpening:
    """Implementation of image sharpening methods."""
    
    @staticmethod
    def unsharp_mask(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                     sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
        """
        Apply unsharp masking for image sharpening.
        
        Args:
            image: Input image array
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian kernel
            amount: Strength of sharpening effect
            
        Returns:
            Sharpened image array
        """
        # Generate the blurred version
        gaussian = cv2.GaussianBlur(image, kernel_size, sigma)
        
        # Calculate the mask
        mask = cv2.subtract(image, gaussian)
        
        # Add weighted mask to original image
        sharpened = cv2.addWeighted(image, 1 + amount, mask, -amount, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def laplacian_sharpening(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply Laplacian filter for edge enhancement.
        
        Args:
            image: Input image array
            kernel_size: Size of the Laplacian kernel
            
        Returns:
            Sharpened image array
        """
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
        
        # Convert back to uint8
        sharpened = image + laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def calculate_edge_strength(image: np.ndarray) -> float:
        """
        Calculate edge strength using Sobel operators.
        
        Args:
            image: Input image array
            
        Returns:
            Edge strength metric
        """
        # Calculate gradients using Sobel
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return mean gradient magnitude as edge strength metric
        return np.mean(gradient_magnitude)