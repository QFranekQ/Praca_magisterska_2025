import cv2
import mediapipe as mp
import numpy as np
import os
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ekstrahuj_cechy_z_filmu(sciezka_filmu):
    """Ekstrahuje punkty kluczowe dłoni oraz rąk (ramion) z filmu."""
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

        # Przetwarzanie dłoni
        results_hands = hands.process(image)
        # Przetwarzanie całego ciała (w tym rąk)
        results_pose = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        frame_features = []

        # Dodaj punkty dłoni (dla maksymalnie 2 dłoni)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                hand_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                frame_features.extend(np.array(hand_points).flatten())
            # Jeśli wykryto tylko 1 dłoń – dodaj zerowe dane dla drugiej
            if len(results_hands.multi_hand_landmarks) == 1:
                frame_features.extend([0] * (21 * 3))
        else:
            # Jeśli brak dłoni – zerowe dane dla obu
            frame_features.extend([0] * (21 * 3 * 2))

        # Dodaj punkty ramion z pose (jeśli są)
        if results_pose.pose_landmarks:
            # Wybrane punkty: lewe i prawe ramię, łokieć, nadgarstek
            indeksy_reki = [11, 13, 15, 12, 14, 16]
            reka_points = [
                [results_pose.pose_landmarks.landmark[i].x,
                 results_pose.pose_landmarks.landmark[i].y,
                 results_pose.pose_landmarks.landmark[i].z]
                for i in indeksy_reki
            ]
            frame_features.extend(np.array(reka_points).flatten())
        else:
            frame_features.extend([0] * (6 * 3))  # 6 punktów po 3 współrzędne

        # Upewnij się, że rozmiar klatki jest stały
        if len(frame_features) == 144:
            cechy.append(frame_features)
        else:
            logging.warning(f"❌ Pominięto klatkę z nieprawidłową liczbą cech: {len(frame_features)}")

    cap.release()
    return cechy

def ekstrahuj_cechy_z_folderu(sciezka_folderu, sciezka_docelowa):
    """Ekstrahuje cechy ze wszystkich filmów w folderze i zapisuje je."""
    for plik in os.listdir(sciezka_folderu):
        if plik.endswith((".mp4", ".avi", ".mkv", ".webm")):
            sciezka_filmu = os.path.join(sciezka_folderu, plik)
            cechy_filmu = ekstrahuj_cechy_z_filmu(sciezka_filmu)
            if cechy_filmu and all(len(f) == 144 for f in cechy_filmu):
                nazwa_pliku = f"{plik.split('.')[0]}.npy"
                sciezka_pliku = os.path.join(sciezka_docelowa, nazwa_pliku)
                np.save(sciezka_pliku, cechy_filmu)
                logging.info(f"✅ Zapisano cechy z: {sciezka_filmu} do: {sciezka_pliku}")
            else:
                logging.warning(f"⚠️ Pominięto plik {plik} — niejednolite lub puste dane.")

def ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa):
    """Ekstrahuje cechy z wszystkich folderów (słów) w danych i zapisuje je."""
    for slowo in os.listdir(sciezka_zrodlowa):
        sciezka_slowa = os.path.join(sciezka_zrodlowa, slowo)
        if os.path.isdir(sciezka_slowa):
            sciezka_docelowa_slowa = os.path.join(sciezka_docelowa, slowo)
            os.makedirs(sciezka_docelowa_slowa, exist_ok=True)
            ekstrahuj_cechy_z_folderu(sciezka_slowa, sciezka_docelowa_slowa)

# Użycie
sciezka_zrodlowa = "valid_dataset"
sciezka_docelowa = "sekwencje_rece"

os.makedirs(sciezka_docelowa, exist_ok=True)
ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa)
