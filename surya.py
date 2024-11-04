import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt

# Median Filter
def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# Bilateral Filter
def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Gaussian Filter
def apply_gaussian_filter(image, kernel_size=(5,5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# AI-based Denoising using pre-trained model
def apply_ai_denoise(image, model_type='fastgan'):
    """
    Apply AI-based denoising using pre-trained models.
    Supports 'fastgan' or 'esrgan' models from TensorFlow Hub.
    """
    # Normalize image to [0, 1]
    image_norm = image.astype(np.float32) / 255.0
    
    try:
        if model_type == 'fastgan':
            # Load FastGAN model (good for denoising)
            model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')
        elif model_type == 'esrgan':
            # Load ESRGAN model (good for enhancement and denoising)
            model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Add batch dimension and process
        image_input = tf.expand_dims(image_norm, axis=0)
        denoised_image = model(image_input)
        
        # Convert back to uint8
        return (np.squeeze(denoised_image, axis=0) * 255).astype(np.uint8)
    except Exception as e:
        print(f"Error in AI denoising: {str(e)}")
        # Fallback to bilateral filter if AI denoising fails
        return apply_bilateral_filter(image)

# Compute SNR (Signal to Noise Ratio)
def compute_snr(clean_image, noisy_image):
    # Ensure images are in float32
    clean = clean_image.astype(np.float32)
    noisy = noisy_image.astype(np.float32)
    
    # Calculate signal power
    signal_power = np.mean(clean ** 2)
    
    # Calculate noise power
    noise = clean - noisy
    noise_power = np.mean(noise ** 2)
    
    # Avoid division by zero
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Process RAW Bayer image
def process_raw_bayer_image(raw_image, wb_gains=(1.0, 1.0, 1.0)):
    # Convert to RGB
    rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GR2RGB)
    
    # Apply white balance
    rgb_balanced = cv2.multiply(rgb_image, np.array(wb_gains))
    
    # Clip values to valid range
    rgb_balanced = np.clip(rgb_balanced, 0, 255).astype(np.uint8)
    
    return rgb_balanced

# Complete ISP pipeline
def isp_pipeline_and_denoise(raw_image_path, model_type='fastgan'):
    """
    Complete image processing pipeline using pre-trained model for denoising.
    """
    print("Starting ISP pipeline processing...")
    
    try:
        # Load RAW image
        raw_image = np.fromfile(raw_image_path, dtype=np.uint16).reshape((1280, 1920))
        print("RAW image loaded successfully")
        
        # Process RAW image with auto white balance
        wb_gains = (1.2, 1.0, 1.8)  # Example gains for tungsten light
        rgb_image = process_raw_bayer_image(raw_image, wb_gains)
        print("RAW processing completed")
        
        # Convert to 8-bit
        rgb_image_8bit = (rgb_image / 16).astype(np.uint8)
        
        # Apply denoising methods
        print("Applying denoising methods...")
        results = {
            'rgb_image': rgb_image_8bit,
            'median_filtered': apply_median_filter(rgb_image_8bit),
            'bilateral_filtered': apply_bilateral_filter(rgb_image_8bit),
            'gaussian_filtered': apply_gaussian_filter(rgb_image_8bit),
            'ai_denoised': apply_ai_denoise(rgb_image_8bit, model_type),
            'edge_strength': compute_gradient_magnitude(rgb_image_8bit)
        }
        print("All processing completed successfully")
        
        return results
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return None

# Compute Gradient Magnitude
def compute_gradient_magnitude(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude

# Visualize results
def visualize_results(results):
    if results is None:
        print("No results to visualize")
        return
        
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original and filtered images
    images = [
        ('Original Image', results['rgb_image']),
        ('Median Filtered', results['median_filtered']),
        ('Bilateral Filtered', results['bilateral_filtered']),
        ('Gaussian Filtered', results['gaussian_filtered']),
        ('AI Denoised', results['ai_denoised']),
        ('Edge Strength', results['edge_strength'])
    ]
    
    for idx, (title, img) in enumerate(images):
        row = idx // 3
        col = idx % 3
        if 'Edge Strength' in title:
            ax[row, col].imshow(img, cmap='gray')
        else:
            ax[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if 'Edge Strength' not in title:  # Don't compute SNR for edge strength
            snr = compute_snr(results['rgb_image'], img)
            ax[row, col].set_title(f'{title}\nSNR: {snr:.2f}dB')
        else:
            ax[row, col].set_title(title)
        ax[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    raw_image_path = "data/input/image.raw"
    model_type = 'fastgan'  # or 'esrgan'
    
    try:
        # Process image
        results = isp_pipeline_and_denoise(raw_image_path, model_type)
        
        # Visualize results
        visualize_results(results)
        
        # Print metrics
        if results:
            print("\nDenoising Performance Metrics:")
            for method in ['median_filtered', 'bilateral_filtered', 'gaussian_filtered', 'ai_denoised']:
                snr = compute_snr(results['rgb_image'], results[method])
                print(f"SNR {method}: {snr:.2f} dB")
                
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()