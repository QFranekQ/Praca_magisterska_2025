import cv2
import os

NOWA_SZEROKOSC = 1920
NOWA_WYSOKOSC = 1080
NOWA_ROZDZIELCZOSC = (NOWA_SZEROKOSC, NOWA_WYSOKOSC)
FPS = 25

sciezka_wejsciowa = "valid_dataset"
sciezka_wyjsciowa = "valid_dataset_skalowane_duze"
os.makedirs(sciezka_wyjsciowa, exist_ok=True)

def skaluj_film(sciezka_wej, sciezka_wyj):
    cap = cv2.VideoCapture(sciezka_wej)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć: {sciezka_wej}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = FPS

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(sciezka_wyj, fourcc, fps, NOWA_ROZDZIELCZOSC)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, NOWA_ROZDZIELCZOSC)
        out.write(frame_resized)

    cap.release()
    out.release()
    print(f"✅ Zapisano: {sciezka_wyj}")

for root, _, pliki in os.walk(sciezka_wejsciowa):
    for plik in pliki:
        if plik.endswith((".mp4", ".avi", ".mkv", ".webm")):
            pelna_sciezka = os.path.join(root, plik)
            rel_path = os.path.relpath(pelna_sciezka, sciezka_wejsciowa)

            # Zmień rozszerzenie na .mp4 niezależnie od oryginału
            nazwa_bez_ext, _ = os.path.splitext(rel_path)
            rel_path_mp4 = nazwa_bez_ext + ".mp4"

            wyjscie_path = os.path.join(sciezka_wyjsciowa, rel_path_mp4)
            os.makedirs(os.path.dirname(wyjscie_path), exist_ok=True)
            skaluj_film(pelna_sciezka, wyjscie_path)
