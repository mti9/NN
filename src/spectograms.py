import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Configuration - CHANGE THESE PATHS
input_folder = 'C:/Users/matko/Desktop/data_h_c_s_a/update_h_s/synthetic_v2'  # Replace with your input folder path
output_folder = 'C:/Users/matko/Desktop/data_h_c_s_a/update_h_s/synthetic_s'  # Replace with your output folder path

# Spectrogram parameters (adjust if needed)
n_fft = 2048
hop_length = 512
n_mels = 128
target_shape = (128, 128)  # All spectrograms will be resized to this

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def generate_spectrogram(input_path, output_path):
    # Load audio file
    y, sr = librosa.load(input_path, sr=None)
    
    # Generate mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
                                      hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Plot and save without axes or decorations
    plt.figure(figsize=(1.28, 1.28), dpi=100)  # Results in 128x128 image
    plt.axis('off')
    plt.tight_layout(pad=0)
    librosa.display.specshow(S_dB, cmap='gray_r')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Process all .wav files in input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, 
                                 os.path.splitext(filename)[0] + '.png')
        generate_spectrogram(input_path, output_path)
        print(f'Processed: {filename}')

print('Spectrogram generation complete!')