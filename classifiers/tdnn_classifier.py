import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import time

# Stałe
DELAY = 3
EPOCHS = 50
BATCH_SIZE = 32

def load_sequences(data_dir):
    sequences = []
    labels = []

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith(".npy"):
                path = os.path.join(label_path, file)
                seq = np.load(path)
                if seq.shape[0] < 5:
                    continue
                sequences.append(seq)
                labels.append(label)

    return sequences, labels

def create_tdnn_data(sequences, delay):
    tdnn_sequences = []
    for seq in sequences:
        if len(seq) <= delay:
            continue
        tdnn_seq = []
        for i in range(delay, len(seq)):
            tdnn_seq.append(seq[i - delay:i].flatten())
        tdnn_sequences.append(np.array(tdnn_seq))
    return tdnn_sequences

def run_classifier(train_dir, test_dir):
    X, y = load_sequences(train_dir)
    X_tdnn_list = create_tdnn_data(X, DELAY)
    max_len = max([len(seq) for seq in X_tdnn_list])
    X_tdnn = pad_sequences(X_tdnn_list, dtype='float32', padding='post', maxlen=max_len)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tdnn, y_encoded, test_size=0.01, random_state=42
    )

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.perf_counter()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.1, verbose=0)
    end_time = time.perf_counter()  
    elapsed_time = end_time - start_time
    X_test_original, y_test_labels = load_sequences(test_dir)
    X_test_tdnn_list = create_tdnn_data(X_test_original, DELAY)
    X_test_tdnn = pad_sequences(X_test_tdnn_list, dtype='float32', padding='post', maxlen=max_len)
    y_test_encoded = le.transform(y_test_labels)

    print("📊 Predykcja...")
    y_pred_probs = model.predict(X_test_tdnn, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test_encoded, y_pred)
    y_test_labels = le.inverse_transform(y_test_encoded)
    y_pred_labels = le.inverse_transform(y_pred)
    return acc,classification_report(y_test_encoded, y_pred,digits=4),y_test_labels, y_pred_labels,elapsed_time
