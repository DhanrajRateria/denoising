import cv2
import numpy as np
from typing import Tuple

class ClassicalDenoising:
    """Implementation of classical denoising methods."""
    
    @staticmethod
    def gaussian_filter(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                       sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian filter for denoising.
        
        Args:
            image: Input image array
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian kernel
            
        Returns:
            Denoised image array
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter for denoising.
        
        Args:
            image: Input image array
            kernel_size: Size of the median filter kernel
            
        Returns:
            Denoised image array
        """
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, 
                        sigma_color: float = 75.0, 
                        sigma_space: float = 75.0) -> np.ndarray:
        """
        Apply bilateral filter for denoising.
        
        Args:
            image: Input image array
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Denoised image array
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def calculate_snr(clean_image: np.ndarray, noisy_image: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR).
        
        Args:
            clean_image: Original clean image
            noisy_image: Noisy or processed image
            
        Returns:
            SNR value in decibels
        """
        # Convert images to float for calculations
        clean_image = clean_image.astype(float)
        noisy_image = noisy_image.astype(float)
        
        # Calculate signal power
        signal_power = np.mean(clean_image ** 2)
        
        # Calculate noise power
        noise = clean_image - noisy_image
        noise_power = np.mean(noise ** 2)
        
        # Calculate SNR
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr