from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import time

def run_classifier(file_path):
    # Wczytaj dane z pliku CSV
    df = pd.read_csv(file_path)

    # Oddziel dane treningowe i testowe na podstawie kolumny 'set'
    train_df = df[df["set"] == "train"].copy()
    test_df = df[df["set"] == "test"].copy()

    # Usuń kolumnę 'set' – niepotrzebna do trenowania
    train_df = train_df.drop(columns=["set"])
    test_df = test_df.drop(columns=["set"])

    # Podziel na cechy i etykiety
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    model =  SVC(kernel='rbf', C=1.0, gamma='scale')
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
  
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,classification_report(y_test, y_pred,digits=4),y_test, y_pred,elapsed_time  
