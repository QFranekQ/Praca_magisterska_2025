import os
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict
from tqdm import tqdm
import time

# Parametry modelu
N_COMPONENTS = 3
COV_TYPE = 'tied'
N_ITER = 300

# Wczytywanie sekwencji z folderów
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

# Trening modeli HMM dla każdej klasy
def train_hmm_models(X_train, y_train):
    models = {}
    label_set = sorted(set(y_train))

    for label in label_set:
        class_sequences = []
        for i in range(len(X_train)):
            if y_train[i] == label:
                class_sequences.append(X_train[i])
        lengths = []
        for seq in class_sequences:
            lengths.append(len(seq))
        X_concat = np.vstack(class_sequences)
        model = hmm.GaussianHMM(n_components=3,
                                covariance_type='tied',
                                n_iter=300,
                                verbose=False)
        try:
            model.fit(X_concat, lengths)
            models[label] = model
            print(f"Wytrenowano model dla '{label}'")
        except Exception as e:
            print(f"Błąd trenowania '{label}': {e}")
    return models

# Predykcja jednej sekwencji
def predict(models, sequence):
    scores = {}
    for label, model in models.items():
        try:
            scores[label] = model.score(sequence)
        except:
            scores[label] = float('-inf')
    return max(scores, key=scores.get)

# Główna funkcja uruchamiana z GUI
def run_classifier(train_dir, test_dir):
    print("📥 Wczytywanie danych treningowych...")
    X_train, y_train = load_sequences(train_dir)

    print("🧠 Trenowanie modeli HMM...")
    start_time = time.perf_counter()
    models = train_hmm_models(X_train, y_train)
    end_time = time.perf_counter()  
    elapsed_time = end_time - start_time
    print("🧪 Wczytywanie danych testowych...")
    X_test, y_test = load_sequences(test_dir)

    y_pred = []
    for seq in tqdm(X_test, desc="📊 Klasyfikacja"):
        pred = predict(models, seq)
        y_pred.append(pred)

    acc = accuracy_score(y_test, y_pred)
    return acc,classification_report(y_test, y_pred,digits=4),y_test, y_pred, elapsed_time
