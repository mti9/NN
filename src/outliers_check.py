import os
import numpy as np
import librosa
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_audio_features(y, sr):
    """Extract features useful for voice detection"""
    features = {}
    
    # Basic audio properties
    features['duration'] = len(y) / sr
    features['rms'] = np.sqrt(np.mean(y**2))  # Root mean square energy
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Harmonic features
    harmonic = librosa.effects.harmonic(y)
    features['harmonic_mean'] = np.mean(harmonic)
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(min(5, mfccs.shape[0])):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
    
    return features

def calculate_outlier_probabilities(scores):
    """Convert outlier scores to probabilities (0-100%)"""
    # Normalize scores to 0-1 range
    min_score, max_score = np.min(scores), np.max(scores)
    normalized = (scores - min_score) / (max_score - min_score)
    
    # Convert to probabilities (lower scores = higher outlier probability)
    probabilities = 100 * (1 - normalized)
    return probabilities

def save_results_to_file(outliers, probabilities, features, output_file="outlier_results.txt"):
    """Save results to a text file with detailed information"""
    with open(output_file, 'w') as f:
        f.write("=== Audio File Outlier Analysis Results ===\n\n")
        f.write(f"Total files analyzed: {len(features)}\n")
        f.write(f"Potential outliers found: {len(outliers)}\n\n")
        
        f.write("=== Outlier Files (Sorted by Probability) ===\n")
        f.write("Filename | Outlier Probability | Duration | Key Features\n")
        f.write("-" * 80 + "\n")
        
        # Sort by probability (descending)
        sorted_files = sorted(zip(outliers, probabilities), 
                           key=lambda x: x[1], reverse=True)
        
        for filename, prob in sorted_files:
            file_features = features[filename]
            f.write(f"{filename} | {prob:.1f}% | {file_features['duration']:.2f}s | ")
            
            # Write most significant features
            if prob > 80:
                reason = "Very low harmonic content" if file_features['harmonic_mean'] < 0.1 else "Abnormal spectral pattern"
            elif prob > 60:
                reason = "Low voice-like features" if file_features['mfcc_0_mean'] > 0 else "Possible noise"
            else:
                reason = "Moderate deviation from typical voice"
                
            f.write(f"{reason}\n")
        
        f.write("\n=== All Files Analysis ===\n")
        f.write("Filename | Outlier Probability | Duration | RMS Energy\n")
        f.write("-" * 80 + "\n")
        for filename in features:
            if filename in outliers:
                prob = probabilities[outliers.index(filename)]
            else:
                prob = 0.0
            f.write(f"{filename} | {prob:.1f}% | {features[filename]['duration']:.2f}s | {features[filename]['rms']:.4f}\n")
        
        f.write("\nAnalysis complete.\n")

def detect_voice_outliers(folder_path, threshold=0.7, plot_results=False):
    """
    Detect audio files that likely don't contain human voice
    
    Returns:
    - List of outlier filenames
    - List of outlier probabilities (0-100%)
    - Dictionary of features for all files
    """
    all_features = []
    filenames = []
    
    # Get list of WAV files
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print("No WAV files found in the directory.")
        return [], [], {}
    
    print(f"Processing {len(wav_files)} audio files...")
    
    # Process each WAV file with progress bar
    for filename in tqdm(wav_files, desc="Analyzing files", unit="file"):
        filepath = os.path.join(folder_path, filename)
        
        try:
            y, sr = librosa.load(filepath, sr=None)
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=0)
            
            # Extract features
            features = extract_audio_features(y, sr)
            features['filename'] = filename
            all_features.append(features)
            filenames.append(filename)
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue
    
    # Convert features to numpy array for analysis
    feature_names = [k for k in all_features[0].keys() if k != 'filename']
    X = np.array([[f[k] for k in feature_names] for f in all_features])
    
    # Normalize features
    X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
    
    print("\nDetecting outliers...")
    # Use Isolation Forest for outlier detection
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_norm)
    scores = clf.decision_function(X_norm)
    probabilities = calculate_outlier_probabilities(scores)
    
    # Determine outliers based on threshold
    is_outlier = probabilities > (threshold * 100)
    
    # Optionally plot feature distributions
    if plot_results:
        plt.figure(figsize=(12, 6))
        for i, name in enumerate(feature_names[:5]):
            plt.subplot(2, 3, i+1)
            plt.hist(X[:,i], bins=20, alpha=0.7, label='All files')
            plt.hist(X[is_outlier,i], bins=20, alpha=0.7, label='Outliers')
            plt.title(name)
            plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Prepare results
    outliers = [filenames[i] for i in range(len(filenames)) if is_outlier[i]]
    outlier_probs = [prob for i, prob in enumerate(probabilities) if is_outlier[i]]
    features_dict = {filenames[i]: all_features[i] for i in range(len(filenames))}
    
    return outliers, outlier_probs, features_dict

if __name__ == "__main__":
    # Configuration
    audio_folder = "C:/Users/matko/Desktop/data_h_c_s_a/human"  # Replace with your folder path
    confidence_threshold = 0.7    # Higher = more strict outlier detection
    output_file = "C:/Users/matko/Desktop/data_h_c_s_a/voice_outlier_analysis.txt"
    
    print("=== Audio Outlier Detection ===")
    print(f"Scanning folder: {audio_folder}")
    
    # Detect outliers
    outliers, probabilities, features = detect_voice_outliers(
        audio_folder, confidence_threshold, plot_results=True)
    
    # Save results to file
    save_results_to_file(outliers, probabilities, features, output_file)
    
    # Print summary
    if outliers:
        print("\n=== Analysis Results ===")
        print(f"Found {len(outliers)} potential outliers out of {len(features)} files.")
        print("Top outliers by probability:")
        for filename, prob in sorted(zip(outliers, probabilities), 
                                  key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {filename} ({prob:.1f}% probability)")
        
        print(f"\nComplete results saved to {output_file}")
    else:
        print("\nNo outlier files detected - all files appear to contain voice.")
        print(f"Analysis summary saved to {output_file}")