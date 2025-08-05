import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from tqdm import tqdm
import time

K = 3  # liczba sąsiadów

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

# Oblicz odległość DTW
def fastdtw_distance(seq1, seq2):
    dist, _ = fastdtw(seq1, seq2, dist=euclidean)
    return dist

# Własna implementacja KNN oparta na FastDTW
def knn_classify(test_seq, train_seqs, train_labels, k=3):
    distances = []
    for ref_seq, label in zip(train_seqs, train_labels):
        dist = fastdtw_distance(test_seq, ref_seq)
        distances.append((dist, label))
    distances.sort(key=lambda x: x[0])   
    k_nearest = distances[:k]
    nearest_labels = []
    for dista, lab in k_nearest:
        nearest_labels.append(lab)
    most_common = Counter(nearest_labels).most_common(1)[0][0]
    return most_common

# Główna funkcja dla GUI
def run_classifier(train_dir, test_dir):
    print("📥 Wczytywanie danych treningowych...")
    train_X, train_y = load_sequences(train_dir)

    print("🧪 Wczytywanie danych testowych...")
    test_X, test_y = load_sequences(test_dir)

    print(f"🔄 Klasyfikacja DTW (k={K})...")
    y_pred = []
    start_time = time.perf_counter()

    for seq in tqdm(test_X, desc="🧠 Klasyfikacja"):
        pred = knn_classify(seq, train_X, train_y, k=K)
        y_pred.append(pred)
    end_time = time.perf_counter()  
    elapsed_time = end_time - start_time
    acc = accuracy_score(test_y, y_pred)
    return acc,classification_report(test_y, y_pred,digits=4),test_y, y_pred, elapsed_time 
