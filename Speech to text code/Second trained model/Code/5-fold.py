import os
import librosa
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import jiwer

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

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=777)
fold_no = 1
histories = []

test_losses = []
test_accuracies = []
wer_scores = []

# Set the starting fold
start_fold = 4

for train_index, val_index in kf.split(all_wave):
    if fold_no < start_fold:
        fold_no += 1
        continue
    
    print(f"Training fold {fold_no}...")
    
    x_train, x_val = all_wave[train_index], all_wave[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
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
    mc = ModelCheckpoint(f'model_fold_{fold_no}.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit(x_train, y_train, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))
    histories.append(history)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=1)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Predict the labels on the validation set
    y_val_pred = model.predict(x_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_true_classes = np.argmax(y_val, axis=1)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_val_true_classes, y_val_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for Fold {fold_no}')
    plt.savefig(f'confusion_matrix_fold_{fold_no}.png')
    plt.close()

    # Calculate WER for each pair of true and predicted labels
    y_true_labels = [classes[i] for i in y_val_true_classes]
    y_pred_labels = [classes[i] for i in y_val_pred_classes]
    fold_wer_scores = [jiwer.wer(true, pred) for true, pred in zip(y_true_labels, y_pred_labels)]
    
    # Calculate the average WER for the fold
    average_wer = np.mean(fold_wer_scores)
    print(f"Average Word Error Rate (WER) for Fold {fold_no}: {average_wer}")
    wer_scores.append(average_wer)

    fold_no += 1

# Aggregate the results from each fold
losses = [history.history['loss'] for history in histories]
val_losses = [history.history['val_loss'] for history in histories]
accuracies = [history.history['accuracy'] for history in histories]
val_accuracies = [history.history['val_accuracy'] for history in histories]

# Plot training & validation loss values for each fold
plt.figure(figsize=(10, 5))
for i in range(len(losses)):
    plt.plot(losses[i], label=f'Training Loss Fold {i+1}')
    plt.plot(val_losses[i], label=f'Validation Loss Fold {i+1}')
plt.title('Model Loss Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')
plt.savefig('model_loss_across_folds.png')
plt.close()

# Plot training & validation accuracy values for each fold
plt.figure(figsize=(10, 5))
for i in range(len(accuracies)):
    plt.plot(accuracies[i], label=f'Training Accuracy Fold {i+1}')
    plt.plot(val_accuracies[i], label=f'Validation Accuracy Fold {i+1}')
plt.title('Model Accuracy Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')
plt.savefig('model_accuracy_across_folds.png')
plt.close()

# Calculate average performance metrics
avg_test_loss = np.mean(val_losses)
avg_test_accuracy = np.mean(val_accuracies)
avg_wer_score = np.mean(wer_scores)

print(f"\nAverage Validation Loss: {avg_test_loss}")
print(f"Average Validation Accuracy: {avg_test_accuracy}")
print(f"Average Word Error Rate (WER): {avg_wer_score}")
