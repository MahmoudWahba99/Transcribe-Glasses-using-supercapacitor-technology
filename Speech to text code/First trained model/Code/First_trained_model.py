import os
import librosa
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Preprocessing steps 
train_audio_path = r"D:\input\train\audio2"
labels = os.listdir(train_audio_path)
all_wave = []
all_label = []

for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
        if len(samples) == 16000:
            all_wave.append(samples)
            all_label.append(label)

# Convert the output labels to integer encoded
le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

# Convert the integer encoded labels to a one-hot vector
y = to_categorical(y, num_classes=len(labels)) 

# Reshape the 2D array to 3D since the input to the conv1d must be a 3D array
all_wave = np.array(all_wave).reshape(-1, 16000, 1)

# Split the data into 70% training and 30% remaining
x_train, x_rem, y_train, y_rem = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.3, random_state=777, shuffle=True)

# Split the remaining 30% into 50% validation and 50% testing (which is 15% of the original dataset each)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, stratify=y_rem, test_size=0.5, random_state=777, shuffle=True)

K.clear_session()
inputs = Input(shape=(16000, 1))

# First Conv1D layer
conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Flatten layer
conv = Flatten()(conv)

# Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

# Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('model0.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(x_train, y_train, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Calculate WER (Word Error Rate)
def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    """
    # Initializing matrix
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i

    # Finding minimum edit distance
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d[len(r)][len(h)] / float(len(r))

# Assuming you have ground truth (y_true) and predicted (y_pred) labels for test data
# Here, I'll use the model to predict the labels on test data
y_pred = model.predict(x_test)
y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
y_true_labels = le.inverse_transform(np.argmax(y_test, axis=1))

# Calculate WER for each pair of true and predicted labels
wer_score = np.mean([wer(true, pred) for true, pred in zip(y_true_labels, y_pred_labels)])
print(f"Word Error Rate (WER): {wer_score}")

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
