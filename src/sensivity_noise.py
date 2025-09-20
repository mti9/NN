import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuration - CHANGE THESE PATHS
input_folder = 'C:/Users/matko/Desktop/data_h_c_s_a/spectograms/train'  # Folder with your spectrograms

# SNR levels to test
snr_levels = [5]  # dB

def add_gaussian_noise(image, snr_db):
    """
    Add Gaussian white noise to image with specified SNR
    """
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Calculate signal power
    signal_power = np.mean(img_array ** 2)
    
    # Calculate noise power based on SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), img_array.shape)
    
    # Add noise to image
    noisy_image = img_array + noise
    
    # Clip to valid range [0, 1] and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    
    return Image.fromarray(noisy_image)

def process_spectrogram_with_noise(input_path, snr_db):
    """
    Process a single spectrogram with noise addition and overwrite original
    """
    try:
        # Load the spectrogram image
        image = Image.open(input_path).convert('L')  # Convert to grayscale
        
        # Add Gaussian noise
        noisy_image = add_gaussian_noise(image, snr_db)
        
        # Overwrite the original image
        noisy_image.save(input_path)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

# Process all .png files in input folder with different SNR levels
for snr_db in snr_levels:
    print(f"Processing with SNR {snr_db}dB...")
    processed_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            
            # Add noise and overwrite
            if process_spectrogram_with_noise(input_path, snr_db):
                processed_count += 1
                if processed_count % 100 == 0:  # Print progress every 100 files
                    print(f'Processed {processed_count} files with SNR {snr_db}dB')
    
    print(f'Completed SNR {snr_db}dB: {processed_count} files processed')

print('Noise addition complete! All original files have been overwritten.')

# Optional: Create a visualization to see the effect
def create_comparison_image(input_folder):
    """
    Create a comparison image showing original vs noisy spectrograms
    """
    # Find example files
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    if not png_files:
        print("No PNG files found for comparison")
        return
    
    # Take first 4 files for comparison
    example_files = png_files[:4]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, filename in enumerate(example_files):
        input_path = os.path.join(input_folder, filename)
        
        try:
            image = Image.open(input_path).convert('L')
            img_array = np.array(image)
            
            axes[i].imshow(img_array, cmap='gray_r')
            axes[i].set_title(f'{filename}\n(After noise addition)')
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading\n{filename}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(input_folder, 'noise_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Comparison image saved: {comparison_path}')

# Create comparison image
create_comparison_image(input_folder)