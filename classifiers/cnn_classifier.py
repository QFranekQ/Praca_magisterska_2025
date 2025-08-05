import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import time

def run_classifier(path):
    # === 1. Wczytanie danych ===
    df = pd.read_csv(path)

    # Oddzielenie zbiorów treningowego i testowego
    train_df = df[df["set"] == "train"].copy()
    test_df = df[df["set"] == "test"].copy()

    # Usunięcie kolumny 'set'
    train_df = train_df.drop(columns=["set"])
    test_df = test_df.drop(columns=["set"])

    # === 2. Przygotowanie danych ===
    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values

    X_test = test_df.drop("label", axis=1).values
    y_test = test_df["label"].values

    # === 3. Etykiety ===
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    num_classes = len(le.classes_)

    # === 4. Reshape do Conv1D (samples, time_steps, features_per_step) ===
    n_features = X_train.shape[1]
    n_coords = 3
    n_landmarks = n_features // n_coords
    X_train = X_train.reshape(-1, n_landmarks, n_coords)
    X_test = X_test.reshape(-1, n_landmarks, n_coords)

    # === 5. Normalizacja ===
    # max_val = np.max(np.abs(X_train))  # Używamy tylko treningowych do normy
    # X_train = X_train / max_val
    # X_test = X_test / max_val

    # === 6. Model ===
    model = models.Sequential([
        layers.Input(shape=(n_landmarks, n_coords)),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # === 7. Trening ===
    start_time = time.perf_counter()
    model.fit(X_train, y_train_encoded, epochs=30, batch_size=32, validation_split=0.1, verbose=0)
    end_time = time.perf_counter()  
    elapsed_time = end_time - start_time
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=le.classes_,digits=4)
    y_test_labels = le.inverse_transform(y_test_encoded)
    y_pred_labels = le.inverse_transform(y_pred)

    return accuracy, report, y_test_labels, y_pred_labels, elapsed_time
