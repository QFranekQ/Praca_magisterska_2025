import cv2
import mediapipe as mp
import numpy as np
import os
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ekstrahuj_cechy_z_filmu(sciezka_filmu):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(sciezka_filmu)
    cechy = []

    if not cap.isOpened():
        logging.error(f"Nie można otworzyć pliku wideo: {sciezka_filmu}")
        return None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                punkty_kluczowe = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
                cechy.append(punkty_kluczowe)

    cap.release()
    return cechy

def ekstrahuj_cechy_z_folderu(sciezka_folderu, sciezka_docelowa):
    for plik in os.listdir(sciezka_folderu):
        if plik.endswith(".mp4") or plik.endswith(".avi") or plik.endswith(".mkv") or plik.endswith(".webm"):  # Obsługa formatów wideo
            sciezka_filmu = os.path.join(sciezka_folderu, plik)
            cechy_filmu = ekstrahuj_cechy_z_filmu(sciezka_filmu)
            if cechy_filmu:
                nazwa_pliku = f"{plik.split('.')[0]}.npy"  # Użyj nazwy pliku wideo jako nazwy pliku .npy
                sciezka_pliku = os.path.join(sciezka_docelowa, nazwa_pliku)
                np.save(sciezka_pliku, cechy_filmu)
                logging.info(f"Zapisano cechy z: {sciezka_filmu} do: {sciezka_pliku}")
            else:
                logging.warning(f"Nie udało się wyodrębnić cech z: {sciezka_filmu}")

def ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa):
    for slowo in os.listdir(sciezka_zrodlowa):
        sciezka_slowa = os.path.join(sciezka_zrodlowa, slowo)
        if os.path.isdir(sciezka_slowa):
            sciezka_docelowa_slowa = os.path.join(sciezka_docelowa, slowo)
            os.makedirs(sciezka_docelowa_slowa, exist_ok=True) # Tworzy folder, jeśli nie istnieje.
            ekstrahuj_cechy_z_folderu(sciezka_slowa, sciezka_docelowa_slowa)

sciezka_zrodlowa = "valid_dataset_skalowane_duze"
sciezka_docelowa = "sekwencje_skalowanie_bduze"

os.makedirs(sciezka_docelowa, exist_ok=True)

ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa)