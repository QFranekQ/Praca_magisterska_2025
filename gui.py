import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import os

# === Importy ===
from classifiers import (
    svm_classifier,
    cnn_classifier,
    knn_classifier,
    dtw_classifier,
    hmm_classifier,
    tdnn_classifier,
)
# === GUI ===
root = tk.Tk()
root.title("🧠 Klasyfikator gestów")
root.geometry("480x550")
style = ttk.Style(root)
style.theme_use("clam")
# === Globalne zmienne ===
train_path = ""
test_path = ""
selected_model = tk.StringVar()
selected_model.set("cnn")  # domyślny wybór

# === Wątki ===
def run_in_thread(target_func, model_name):
    def threaded():
        try:
            acc = target_func(train_path, test_path)
            messagebox.showinfo("Wynik", f"Dokładność ({model_name}): {acc:.2%}")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            run_button.config(state=tk.NORMAL)
    threading.Thread(target=threaded, daemon=True).start()
    run_button.config(state=tk.DISABLED)

# === Obsługa folderów ===
def select_train_folder():
    global train_path
    path = filedialog.askdirectory(title="Wybierz folder treningowy")
    if path:
        train_path = path
        train_label.config(text=os.path.basename(path))

def select_test_folder():
    global test_path
    path = filedialog.askdirectory(title="Wybierz folder testowy")
    if path:
        test_path = path
        test_label.config(text=os.path.basename(path))

# === Start klasyfikatora ===
def start_classification():
    model_key = selected_model.get()
    model_map = {
        "svm": (svm_classifier.run_classifier, "SVM"),
        "cnn": (cnn_classifier.run_classifier, "CNN"),
        "knn": (knn_classifier.run_classifier, "KNN"),
        "tdnn": (tdnn_classifier.run_classifier, "TDNN"),
        "hmm": (hmm_classifier.run_classifier, "HMM"),
        "dtw": (dtw_classifier.run_classifier, "DTW"),
    }
    if train_path == "" or test_path == "":
        messagebox.showwarning("Brak danych", "Wybierz folder treningowy i testowy.")
        return
    func, name = model_map[model_key]
    run_in_thread(func, name)



# === Wybór folderów ===
ttk.Button(root, text="📂 Folder treningowy", command=select_train_folder).pack(pady=5)
train_label = ttk.Label(root, text="(nie wybrano)", foreground="gray")
train_label.pack()

ttk.Button(root, text="📂 Folder testowy", command=select_test_folder).pack(pady=5)
test_label = ttk.Label(root, text="(nie wybrano)", foreground="gray")
test_label.pack()

# === Modele — Radiobuttony ===
ttk.Label(root, text="\n📸 Modele dla danych obrazowych", font=("Arial", 11, "bold")).pack(pady=(10, 0))

ttk.Radiobutton(root, text="CNN (Conv1D)", variable=selected_model, value="cnn").pack(anchor=tk.W, padx=20)
ttk.Radiobutton(root, text="SVM", variable=selected_model, value="svm").pack(anchor=tk.W, padx=20)
ttk.Radiobutton(root, text="KNN", variable=selected_model, value="knn").pack(anchor=tk.W, padx=20)

ttk.Label(root, text="\n🔁 Modele sekwencyjne (czasowe)", font=("Arial", 11, "bold")).pack(pady=(10, 0))

ttk.Radiobutton(root, text="TDNN", variable=selected_model, value="tdnn").pack(anchor=tk.W, padx=20)
ttk.Radiobutton(root, text="HMM", variable=selected_model, value="hmm").pack(anchor=tk.W, padx=20)
ttk.Radiobutton(root, text="DTW + KNN", variable=selected_model, value="dtw").pack(anchor=tk.W, padx=20)

# === Uruchomienie ===
run_button = ttk.Button(root, text="🚀 Uruchom klasyfikację", command=start_classification)
run_button.pack(pady=20)

root.mainloop()
