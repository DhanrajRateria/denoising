import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from denoising.classical import ClassicalDenoising
from denoising.deep_learning import UNetDenoiser
from sharpening.edge_enhancement import ImageSharpening
from utils.image_utils import ImageUtils
import config

def process_image(image_path: str, output_dir: str):
    """Process a single image with all methods."""
    
    # Initialize classes
    denoiser = ClassicalDenoising()
    deep_denoiser = UNetDenoiser()
    sharpener = ImageSharpening()
    utils = ImageUtils()
    
    # Load image
    original = utils.load_image(image_path)
    if original is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Create noisy versions
    noisy_gaussian = utils.add_noise(original, 'gaussian', 
                                   config.NOISE_MEAN, config.NOISE_VAR)
    noisy_sp = utils.add_noise(original, 'salt_pepper')
    
    # Apply denoising methods
    denoised_gaussian = denoiser.gaussian_filter(noisy_gaussian, 
                                               config.GAUSSIAN_KERNEL_SIZE,
                                               config.GAUSSIAN_SIGMA)
    denoised_median = denoiser.median_filter(noisy_sp, 
                                           config.MEDIAN_KERNEL_SIZE)
    denoised_bilateral = denoiser.bilateral_filter(noisy_gaussian,
                                                 config.BILATERAL_D,
                                                 config.BILATERAL_SIGMA_COLOR,
                                                 config.BILATERAL_SIGMA_SPACE)
    
    # Apply sharpening methods
    sharpened_unsharp = sharpener.unsharp_mask(original,
                                              config.GAUSSIAN_KERNEL_SIZE,
                                              config.GAUSSIAN_SIGMA,
                                              config.UNSHARP_AMOUNT)
    sharpened_laplacian = sharpener.laplacian_sharpening(original,
                                                        config.LAPLACIAN_KERNEL_SIZE)
    
    # Calculate metrics
    results = {
        'gaussian_snr': denoiser.calculate_snr(original, denoised_gaussian),
        'median_snr': denoiser.calculate_snr(original, denoised_median),
        'bilateral_snr': denoiser.calculate_snr(original, denoised_bilateral),
        'unsharp_edge_strength': sharpener.calculate_edge_strength(sharpened_unsharp),
        'laplacian_edge_strength': sharpener.calculate_edge_strength(sharpened_laplacian)
    }
    
    # Save results
    base_name = Path(image_path).stem
    utils.save_image(noisy_gaussian, os.path.join(output_dir, f"{base_name}_noisy_gaussian.png"))
    utils.save_image(noisy_sp, os.path.join(output_dir, f"{base_name}_noisy_sp.png"))
    utils.save_image(denoised_gaussian, os.path.join(output_dir, f"{base_name}_denoised_gaussian.png"))
    utils.save_image(denoised_median, os.path.join(output_dir, f"{base_name}_denoised_median.png"))
    utils.save_image(denoised_bilateral, os.path.join(output_dir, f"{base_name}_denoised_bilateral.png"))
    utils.save_image(sharpened_unsharp, os.path.join(output_dir, f"{base_name}_sharpened_unsharp.png"))
    utils.save_image(sharpened_laplacian, os.path.join(output_dir, f"{base_name}_sharpened_laplacian.png"))
    
    # Create comparison plot
    utils.plot_comparison(
        [original, noisy_gaussian, denoised_gaussian, denoised_bilateral,
         sharpened_unsharp, sharpened_laplacian],
        ['Original', 'Noisy', 'Gaussian', 'Bilateral', 'Unsharp', 'Laplacian']
    )
    
    return results

def main():
    # Process all images in input directory
    for image_file in tqdm(os.listdir(config.INPUT_DIR)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(config.INPUT_DIR, image_file)
            results = process_image(image_path, config.OUTPUT_DIR)
            
            if results:
                print(f"\nResults for {image_file}:")
                for metric, value in results.items():
                    print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()