import cv2
import mediapipe as mp
import numpy as np
import os
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ekstrahuj_cechy_z_filmu(sciezka_filmu):
    """Ekstrahuje znormalizowane cechy dłoni z filmu wraz z pozycją dłoni (centroidem)."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(sciezka_filmu)
    cechy = []

    if not cap.isOpened():
        logging.error(f"❌ Nie można otworzyć pliku wideo: {sciezka_filmu}")
        return None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Domyślne brakujące dłonie
        left = None
        right = None
        left_center = np.zeros(2)
        right_center = np.zeros(2)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                # Oblicz centroid
                center = np.mean(keypoints[:, :2], axis=0)

                # Normalizuj względem centroidu
                keypoints[:, :2] -= center

                if handedness == "Left":
                    left = keypoints
                    left_center = center
                else:
                    right = keypoints
                    right_center = center

        # Jeśli nie wykryto jednej z dłoni
        if left is None:
            left = np.zeros((21, 3))
            left_center = np.zeros(2)
        if right is None:
            right = np.zeros((21, 3))
            right_center = np.zeros(2)

        # 🔧 Połącz cechy z obu dłoni i centroidy w jeden wektor
        frame_features = np.concatenate([
            left.flatten(),
            right.flatten(),
            left_center.flatten(),
            right_center.flatten()
        ])
        cechy.append(frame_features)

    cap.release()
    return cechy

def ekstrahuj_cechy_z_folderu(sciezka_folderu, sciezka_docelowa):
    """Ekstrahuje cechy z filmów w folderze i zapisuje jako .npy."""
    for plik in os.listdir(sciezka_folderu):
        if plik.endswith((".mp4", ".avi", ".mkv", ".webm")):
            sciezka_filmu = os.path.join(sciezka_folderu, plik)
            cechy_filmu = ekstrahuj_cechy_z_filmu(sciezka_filmu)
            if cechy_filmu:
                nazwa_pliku = f"{plik.split('.')[0]}.npy"
                sciezka_pliku = os.path.join(sciezka_docelowa, nazwa_pliku)
                np.save(sciezka_pliku, np.array(cechy_filmu))
                logging.info(f"✅ Zapisano cechy z: {sciezka_filmu} do: {sciezka_pliku}")
            else:
                logging.warning(f"⚠️ Nie udało się wyodrębnić cech z: {sciezka_filmu}")

def ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa):
    """Ekstrahuje cechy ze wszystkich folderów słów."""
    for slowo in os.listdir(sciezka_zrodlowa):
        sciezka_slowa = os.path.join(sciezka_zrodlowa, slowo)
        if os.path.isdir(sciezka_slowa):
            sciezka_docelowa_slowa = os.path.join(sciezka_docelowa, slowo)
            os.makedirs(sciezka_docelowa_slowa, exist_ok=True)
            ekstrahuj_cechy_z_folderu(sciezka_slowa, sciezka_docelowa_slowa)

# 🔧 Ścieżki do danych
sciezka_zrodlowa = "valid_dataset_skalowane_duze"
sciezka_docelowa = "sekwencje_z_centroidem"

os.makedirs(sciezka_docelowa, exist_ok=True)

# 🔁 Start ekstrakcji
ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa)
