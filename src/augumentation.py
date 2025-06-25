import os
import glob
import librosa
import soundfile as sf
import random

# Set your folder path
folder = r'C:/Users/matko/Desktop/data_h_c_s_a/update_h_s/wavs_v2'
target_count = 12419

# Get all .wav files (excluding already augmented ones)
wav_files = [f for f in glob.glob(os.path.join(folder, '*.wav')) if '_SA' not in os.path.basename(f)]
existing_files = len(glob.glob(os.path.join(folder, '*.wav')))

augment_index = 1

while existing_files < target_count:
    for file in wav_files:
        if existing_files >= target_count:
            break
        y, sr = librosa.load(file, sr=None)
        # Randomly choose a pitch shift between -3 and +3 semitones (excluding 0)
        n_steps = random.choice([-3, -2, -1, 1, 2, 3])
        y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
        # Create new filename
        base = os.path.splitext(os.path.basename(file))[0]
        new_name = f"{base}_{augment_index}_SA.wav"
        new_path = os.path.join(folder, new_name)
        sf.write(new_path, y_shifted, sr)
        existing_files += 1
        augment_index += 1

print("Augmentation complete.")