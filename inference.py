# Necessary imports
import joblib
import librosa
import numpy as np

# Load the pre-trained model and LabelEncoder locally
loaded_model = joblib.load('model.joblib')
loaded_le = joblib.load('label.joblib')

# Set default parameters for feature extraction
n_fft = 2048
hop_length = 512
win_length = 2048
window = 'hann'
n_mels = 128
n_bands = 6
fmin = 200

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        # Load audio file and extract features
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', n_mels=n_mels).T, axis=0)
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_bands=n_bands, fmin=fmin).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
        
        # Combine all features into one feature vector
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        return features
    except Exception as e:
        print(f"Error: Exception occurred in feature extraction - {e}")
        return None

# Function to predict the cry type
def predict_cry(file_path):
    # Extract features from the new audio file
    features = extract_features(file_path)
    
    if features is not None:
        # Reshape features to match the input shape expected by the model
        features = features.reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = loaded_model.predict(features)
        
        # Convert prediction back to original label using LabelEncoder
        predicted_label = loaded_le.inverse_transform(prediction)
        
        return predicted_label[0]
    else:
        return "Error: Could not extract features from the audio file"

# Example usage
file_path = 'path/to/your/file.wav'  # Replace with the actual file path
result = predict_cry(file_path)
print(f"Predicted cry type: {result}")
