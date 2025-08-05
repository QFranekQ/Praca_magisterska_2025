import cv2
import mediapipe as mp
import numpy as np
import os
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalizuj_dlon(punkty):
    """Normalizuje punkty dłoni względem jej środka i rozpiętości."""
    punkty = np.array(punkty)
    srodek = np.mean(punkty, axis=0)
    rozmiar = np.linalg.norm(np.max(punkty, axis=0) - np.min(punkty, axis=0))
    if rozmiar == 0:
        rozmiar = 1.0  # zabezpieczenie
    return (punkty - srodek) / rozmiar

def ekstrahuj_cechy_z_filmu(sciezka_filmu):
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                      refine_landmarks=True,
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
        
        results_hands = hands.process(image)
        results_face = face_mesh.process(image)

        frame_features = []

        broda_punkt = None

        if results_face.multi_face_landmarks:
            # Wybieramy punkt 152 - środek dolnej brody
            broda_landmark = results_face.multi_face_landmarks[0].landmark[152]
            broda_punkt = np.array([broda_landmark.x, broda_landmark.y, broda_landmark.z])

        if results_hands.multi_hand_landmarks:
            left = right = None
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                hand_label = handedness.classification[0].label
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                if broda_punkt is not None:
                    # Odejmiemy pozycję brody od każdego punktu dłoni
                    landmarks = landmarks - broda_punkt

                normalized = normalizuj_dlon(landmarks)

                if hand_label == "Left":
                    left = normalized
                else:
                    right = normalized

            # Jeśli którejś dłoni brak, wstawiamy wektor zerowy
            if left is None:
                left = np.zeros((21, 3))
            if right is None:
                right = np.zeros((21, 3))

            frame_features = np.concatenate([left, right], axis=0).flatten()
            cechy.append(frame_features)

    cap.release()
    return cechy if cechy else None

def ekstrahuj_cechy_z_folderu(sciezka_folderu, sciezka_docelowa):
    for plik in os.listdir(sciezka_folderu):
        if plik.endswith((".mp4", ".avi", ".mkv", ".webm")):
            sciezka_filmu = os.path.join(sciezka_folderu, plik)
            cechy_filmu = ekstrahuj_cechy_z_filmu(sciezka_filmu)
            if cechy_filmu:
                try:
                    cechy_array = np.array(cechy_filmu)
                    nazwa_pliku = f"{plik.split('.')[0]}.npy"
                    sciezka_pliku = os.path.join(sciezka_docelowa, nazwa_pliku)
                    np.save(sciezka_pliku, cechy_array)
                    logging.info(f"✅ Zapisano cechy z: {sciezka_filmu} do: {sciezka_pliku}")
                except Exception as e:
                    logging.error(f"Błąd przy zapisie {plik}: {e}")
            else:
                logging.warning(f"⚠️ Nie udało się wyodrębnić cech z: {sciezka_filmu}")

def ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa):
    for slowo in os.listdir(sciezka_zrodlowa):
        sciezka_slowa = os.path.join(sciezka_zrodlowa, slowo)
        if os.path.isdir(sciezka_slowa):
            sciezka_docelowa_slowa = os.path.join(sciezka_docelowa, slowo)
            os.makedirs(sciezka_docelowa_slowa, exist_ok=True)
            ekstrahuj_cechy_z_folderu(sciezka_slowa, sciezka_docelowa_slowa)

# Ścieżki
sciezka_zrodlowa = "valid_dataset_skalowane_duze"
sciezka_docelowa = "sekwencje_skalowanie_norm_broda"

os.makedirs(sciezka_docelowa, exist_ok=True)

if __name__ == "__main__":
    ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa)
