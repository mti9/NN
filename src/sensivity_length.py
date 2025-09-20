import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# CONFIGURATION - CHANGE THESE PATHS
input_folder = 'C:/Users/matko/Desktop/data_h_c_s_a/update_h_s/synthetic_s'  # Folder with your 5s spectrograms
output_folder = input_folder  # Save in the same folder (overwrite) OR change to a new path

# Target lengths in seconds (from 5s base)
target_lengths_seconds = [1, 2, 3, 10]  # List of lengths you want to generate

# Derived parameters (DO NOT CHANGE if your original specs were as above)
base_length_seconds = 5
original_width = 128 px # Width of your current spectrograms (time axis)
hop_length_ms = (base_length_seconds * 1000) / original_width # ms per pixel
# For 5s/128px = 39.0625 ms per pixel

def create_spectrogram_variant(img_array, target_sec, base_sec=5, original_width=128):
    """
    Creates a new spectrogram image array for a target duration.
    For shorter durations: temporal cropping based on energy.
    For longer durations: looping from the start.
    """
    current_width = img_array.shape[1]
    target_width_pixels = int((target_sec / base_sec) * original_width)
    
    # For shorter samples: CROP
    if target_sec < base_sec:
        # Calculate energy per time column (sum of pixel values in the column)
        energy_per_column = np.sum(img_array, axis=(0))
        # Find the center of energy (avoiding very edges)
        center_of_energy = np.argmax(energy_per_column[5:-5]) + 5 # Avoid edge artifacts
        # Calculate start index for cropping to center the most energetic part
        start_pixel = max(0, center_of_energy - target_width_pixels // 2)
        # Ensure we don't crop beyond the image boundary
        start_pixel = min(start_pixel, original_width - target_width_pixels)
        # Perform the crop
        cropped_img = img_array[:, start_pixel:start_pixel + target_width_pixels]
        # Resize back to 128x128 using PIL's LANCZOS (high-quality) resampling
        pil_img = Image.fromarray(cropped_img)
        resized_img = pil_img.resize((original_width, original_width), Image.Resampling.LANCZOS)
        return np.array(resized_img)

    # For longer samples: LOOP
    elif target_sec > base_sec:
        # Calculate how many times we need to loop and the remainder
        num_full_loops = target_width_pixels // original_width
        remainder_pixels = target_width_pixels % original_width
        
        # Start with the original image
        looped_img = np.copy(img_array)
        # Append full loops
        for _ in range(num_full_loops - 1):
            looped_img = np.concatenate((looped_img, img_array), axis=1)
        # Append the necessary part from the beginning for the remainder
        if remainder_pixels > 0:
            looped_img = np.concatenate((looped_img, img_array[:, :remainder_pixels]), axis=1)
        
        # Resize the long image back down to 128px width (compressing the time axis)
        pil_img = Image.fromarray(looped_img)
        # Width is now target_width_pixels, height is 128. Resize to 128x128.
        resized_img = pil_img.resize((original_width, original_width), Image.Resampling.LANCZOS)
        return np.array(resized_img)

    # If target is 5s, return the original
    else:
        return img_array

# Process for each target length
for target_sec in target_lengths_seconds:
    print(f"\nProcessing target length: {target_sec}s")
    
    # Create a subfolder for each length if saving separately
    # length_output_folder = os.path.join(output_folder, f'{target_sec}s_spectrograms')
    # os.makedirs(length_output_folder, exist_ok=True)
    # Using the same folder for overwrite

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            
            # Load the image as grayscale array
            img = Image.open(input_path).convert('L')
            img_array = np.array(img)
            
            # Create the spectrogram variant
            new_spectrogram = create_spectrogram_variant(img_array, target_sec)
            
            # Convert back to PIL Image and save
            output_img = Image.fromarray(new_spectrogram.astype(np.uint8))
            
            # Save in the same folder with the same name (overwrite)
            # OR with a new name to preserve originals:
            # new_filename = f"{os.path.splitext(filename)[0]}_{target_sec}s.png"
            # output_path = os.path.join(output_folder, new_filename)
            
            output_path = os.path.join(output_folder, filename) # Overwrite original
            output_img.save(output_path)
            
            print(f'Converted: {filename} -> {target_sec}s')

print('\nAll spectrogram conversions complete!')