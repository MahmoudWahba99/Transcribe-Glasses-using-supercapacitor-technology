import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import confusion_matrix
import jiwer
import seaborn as sns
def add_noise(samples, noise_factor=0.005):
    noise = np.random.randn(len(samples))
    augmented_data = samples + noise_factor * noise
    augmented_data = augmented_data.astype(type(samples[0]))
    return augmented_data

def shift_time(samples, shift_max=2, shift_direction='both'):
    shift = np.random.randint(low=-shift_max * 1000, high=shift_max * 1000)
    if shift_direction == 'right':
        shift = abs(shift)
    elif shift_direction == 'left':
        shift = -abs(shift)
    elif shift_direction == 'both':
        pass
    augmented_data = np.roll(samples, shift)
    return augmented_data

train_audio_path = r"D:\input\train\audio2"
labels = os.listdir(train_audio_path)
all_features = []
all_label = []

for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
        
        # Augmentation: original, noise-added, time-shifted
        augmented_samples = [
            samples,
            add_noise(samples),
            shift_time(samples)
        ]
        
        for samples_aug in augmented_samples:
            mfcc = librosa.feature.mfcc(y=samples_aug, sr=16000, n_mfcc=40)
            if mfcc.shape[1] == 32:  # Ensure consistent shape
                all_features.append(mfcc)
                all_label.append(label)

# Convert features and labels to numpy arrays
all_features = np.array(all_features)
all_features = all_features.reshape(all_features.shape[0], all_features.shape[1], all_features.shape[2], 1)
all_label = np.array(all_label)

# Convert the output labels to integer encoded
le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

# Convert the integer encoded labels to a one-hot vector
y = to_categorical(y, num_classes=len(labels))

# Train the model on 80% of the data and validate on the remaining 20%
# Split the data into 70% training and 30% remaining
x_train, x_rem, y_train, y_rem = train_test_split(all_features, y, stratify=y, test_size=0.3, random_state=777, shuffle=True)

# Split the remaining 30% into 50% validation and 50% testing (which is 15% of the original dataset each)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, stratify=y_rem, test_size=0.5, random_state=777, shuffle=True)

K.clear_session()
inputs = Input(shape=(40, 32, 1))

def print_shape(layer_output, layer_name):
    print(f"Shape after {layer_name}: {layer_output.shape}")

# First Conv2D layer
conv = Conv2D(16, (3, 3), padding='same', activation='relu', strides=1)(inputs)
conv = BatchNormalization()(conv)
conv = MaxPooling2D(pool_size=(2, 2))(conv)
print_shape(conv, 'First Conv2D and MaxPooling')
conv = Dropout(0.2)(conv)

# Second Conv2D layer
conv = Conv2D(32, (3, 3), padding='same', activation='relu', strides=1)(conv)
conv = BatchNormalization()(conv)
conv = MaxPooling2D(pool_size=(2, 2))(conv)
print_shape(conv, 'Second Conv2D and MaxPooling')
conv = Dropout(0.2)(conv)

# Third Conv2D layer
conv = Conv2D(64, (3, 3), padding='same', activation='relu', strides=1)(conv)
conv = BatchNormalization()(conv)
conv = MaxPooling2D(pool_size=(2, 2))(conv)
print_shape(conv, 'Third Conv2D and MaxPooling')
conv = Dropout(0.2)(conv)

# Fourth Conv2D layer
conv = Conv2D(128, (3, 3), padding='same', activation='relu', strides=1)(conv)
conv = BatchNormalization()(conv)
conv = MaxPooling2D(pool_size=(2, 2))(conv)
print_shape(conv, 'Fourth Conv2D and MaxPooling')
conv = Dropout(0.2)(conv)

# Fifth Conv2D layer
conv = Conv2D(256, (3, 3), padding='same', activation='relu', strides=1)(conv)
conv = BatchNormalization()(conv)
conv = MaxPooling2D(pool_size=(2, 2))(conv)
print_shape(conv, 'Fifth Conv2D and MaxPooling')
conv = Dropout(0.2)(conv)

# Flatten layer
conv = Flatten()(conv)
print_shape(conv, 'Flatten')

# Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.2)(conv)

# Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.2)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('model3.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(x_train, y_train, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict the labels on the test set
y_pred = model.predict(x_test)
y_test_pred_classes = np.argmax(y_pred, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test_true_classes, y_test_pred_classes)

# Calculate WER for each pair of true and predicted labels
y_true_labels = [classes[i] for i in y_test_true_classes]
y_pred_labels = [classes[i] for i in y_test_pred_classes]
wer_scores = []
for true, pred in zip(y_true_labels, y_pred_labels):
    wer_score = jiwer.wer(true, pred)
    wer_scores.append(wer_score)

# Calculate the average WER
average_wer = np.mean(wer_scores)
print(f"Average Word Error Rate (WER): {average_wer}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
