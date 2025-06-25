import os
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import librosa

def find_min_silence_cut(audio, sr, target_length):
    """Find the best place to cut audio based on minimum silence"""
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Calculate short-term energy
    frame_length = int(0.02 * sr)  # 20ms frames
    hop_length = frame_length // 2
    energy = np.array([
        np.sum(np.abs(audio[i:i+frame_length]**2))
        for i in range(0, len(audio)-frame_length, hop_length)
    ])
    
    # Normalize energy
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
    
    # Find silence threshold (10% of max energy)
    threshold = 0.1
    silent_frames = energy < threshold
    
    # Find all possible cut points (5 seconds from start)
    target_samples = int(target_length * sr)
    max_possible_start = len(audio) - target_samples
    
    if max_possible_start <= 0:
        return 0  # Audio is shorter than target
    
    # Find the cut point with most silent frames in the surrounding area
    best_cut = 0
    min_silence_around = float('inf')
    
    # Check every 0.1 second for cut points
    for cut_candidate in range(0, max_possible_start, int(0.1 * sr)):
        # Check 0.5s window around cut point
        window_start = max(0, cut_candidate - int(0.25 * sr))
        window_end = min(len(silent_frames), cut_candidate + int(0.25 * sr))
        silence_count = np.sum(silent_frames[window_start:window_end])
        
        if silence_count < min_silence_around:
            min_silence_around = silence_count
            best_cut = cut_candidate
    
    return best_cut

def process_audio_file(input_path, output_path, target_length=5):
    # Load audio file
    audio, sr = librosa.load(input_path, sr=None, mono=False)
    
    # Get current length in seconds
    current_length = len(audio) / sr
    
    if current_length > target_length:
        # Audio is too long - find best place to cut
        cut_point = find_min_silence_cut(audio, sr, target_length)
        target_samples = int(target_length * sr)
        processed_audio = audio[cut_point:cut_point+target_samples]
    elif current_length < target_length:
        # Audio is too short - loop the beginning
        needed_samples = int(target_length * sr) - len(audio)
        repeat_part = audio[:needed_samples]
        
        if len(audio.shape) == 1:  # Mono
            processed_audio = np.concatenate([audio, repeat_part])
        else:  # Stereo
            processed_audio = np.concatenate([audio, repeat_part], axis=1)
    else:
        # Audio is exactly 5 seconds
        processed_audio = audio
    
    # Save processed audio
    if len(processed_audio.shape) == 1:
        sf.write(output_path, processed_audio, sr, subtype='PCM_16')
    else:
        sf.write(output_path, processed_audio.T, sr, subtype='PCM_16')

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all WAV files in input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Processing {filename}...")
            try:
                process_audio_file(input_path, output_path)
                print(f"Saved processed file to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_folder = "C:/Users/matko/Desktop/data_h_c_s_a/n_data/anonymized"  # Replace with your input folder path
    output_folder = "C:/Users/matko/Desktop/data_h_c_s_a/pp_a"  # Replace with your output folder path
    
    process_folder(input_folder, output_folder)
    print("Processing complete!")