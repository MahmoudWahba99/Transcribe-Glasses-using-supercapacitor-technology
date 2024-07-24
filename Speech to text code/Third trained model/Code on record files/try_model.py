import os
import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from jiwer import wer

# Load the trained model
model = load_model('model3.keras')

# Load the label encoder
train_audio_path = r"D:\input\train\audio2"
labels = os.listdir(train_audio_path)
le = LabelEncoder()
le.fit(labels)
classes = list(le.classes_)

# Function to preprocess new audio recordings
def preprocess_audio(file_path):
    samples, sample_rate = librosa.load(file_path, sr=16000)
    
    # Calculate the MFCC features
    mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=40)
    
    # Pad or trim MFCC to ensure it has 32 frames
    if mfcc.shape[1] < 32:
        pad_width = 32 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > 32:
        mfcc = mfcc[:, :32]
    
    # Reshape the MFCC to the required input shape for the model
    mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    return mfcc

# Function to make predictions on new audio recordings
def predict_audio(file_path):
    mfcc = preprocess_audio(file_path)
    if mfcc is not None:
        prediction = model.predict(mfcc)
        predicted_label = classes[np.argmax(prediction)]
        return predicted_label, mfcc
    else:
        return "Error: Audio file does not have the correct shape", None

# Calculate the average WER and accuracy for multiple audio files
def calculate_average_wer_and_accuracy(audio_files, ground_truth_labels):
    total_wer = 0.0
    valid_predictions = 0
    correct_predictions = 0

    for file_path, ground_truth in zip(audio_files, ground_truth_labels):
        predicted_label, _ = predict_audio(file_path)
        if predicted_label != "Error: Audio file does not have the correct shape":
            error_rate = wer(ground_truth, predicted_label)
            total_wer += error_rate
            valid_predictions += 1
            if predicted_label == ground_truth:
                correct_predictions += 1
            print(f"File: {file_path}")
            print(f"Transcription: {predicted_label}")
            print(f"Ground Truth: {ground_truth}")
            print(f"WER: {error_rate}\n")
        else:
            print(f"Skipping file {file_path} due to shape error")

    if valid_predictions > 0:
        average_wer = total_wer / valid_predictions
        accuracy = (correct_predictions / valid_predictions) * 100
    else:
        average_wer = float('inf')  # No valid predictions
        accuracy = 0.0

    return average_wer, accuracy

# Example usage
new_audio_files = [
     r"c:\Users\hp\Downloads\labels\bed\bed.wav",
    r"c:\Users\hp\Downloads\labels\bird\bird.wav",
    r"c:\Users\hp\Downloads\labels\cat\cat.wav",
    r"c:\Users\hp\Downloads\labels\dog\dog.wav",
    r"c:\Users\hp\Downloads\labels\down\down.wav",
    r"c:\Users\hp\Downloads\labels\eight\eight.wav",
    r"c:\Users\hp\Downloads\labels\four\four.wav",
    r"c:\Users\hp\Downloads\labels\five\five.wav",
    r"c:\Users\hp\Downloads\labels\go\go.wav",
    r"c:\Users\hp\Downloads\labels\happy\happy.wav",
    r"c:\Users\hp\Downloads\labels\house\house.wav",
    r"c:\Users\hp\Downloads\labels\left\left.wav",
    r"c:\Users\hp\Downloads\labels\marvin\marvin.wav",
    r"c:\Users\hp\Downloads\labels\nine\nine.wav",
    r"c:\Users\hp\Downloads\labels\no\no.wav",
    r"c:\Users\hp\Downloads\labels\off\off.wav",
    r"c:\Users\hp\Downloads\labels\on\on.wav",
    r"c:\Users\hp\Downloads\labels\one\one.wav",
    r"c:\Users\hp\Downloads\labels\right\right.wav",
    r"c:\Users\hp\Downloads\labels\seven\seven.wav",
    r"c:\Users\hp\Downloads\labels\sheila\sheila.wav",
    r"c:\Users\hp\Downloads\labels\six\six.wav",
    r"c:\Users\hp\Downloads\labels\stop\stop.wav",
    r"c:\Users\hp\Downloads\labels\three\three.wav",
    r"c:\Users\hp\Downloads\labels\tree\tree.wav",
    r"c:\Users\hp\Downloads\labels\two\two.wav",
    r"c:\Users\hp\Downloads\labels\up\up.wav",
    r"c:\Users\hp\Downloads\labels\wow\wow.wav",
    r"c:\Users\hp\Downloads\labels\yes\yes.wav",
    r"c:\Users\hp\Downloads\labels\zero\zero.wav",
]

ground_truth_labels = [
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "four",
    "five",
    "go",
    "happy",
    "house",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "wow",
    "yes",
    "zero",
]

average_wer, accuracy = calculate_average_wer_and_accuracy(new_audio_files, ground_truth_labels)
print(f"Average WER: {average_wer}")
print(f"Accuracy: {accuracy}%")
