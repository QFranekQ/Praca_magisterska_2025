import os
import cv2
import numpy as np
import logging
from mmpose.apis import init_pose_model, inference_bottom_up_pose_model

# Konfiguracja logowania
tlogging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ścieżki do konfiguracji i wag modelu bottom-up WholeBody
POSE_CONFIG = 'configs/body/2d_kpt_sview_rgb_img/bottom_up/higherhrnet_w48_coco_wholebody_512x512.py'
POSE_CHECKPOINT = 'checkpoints/higherhrnet_w48_coco_wholebody_512x512-3dc56c10.pth'

def ekstrahuj_cechy_z_filmu(sciezka_filmu, pose_model):
    """Ekstrahuje punkty kluczowe dłoni z filmu za pomocą MMPose bottom-up WholeBody."""
    cap = cv2.VideoCapture(sciezka_filmu)
    cechy = []

    if not cap.isOpened():
        logging.error(f"Nie można otworzyć pliku wideo: {sciezka_filmu}")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference bottom-up (zwraca listę osób z kluczowymi punktami 133x3)
        pose_results = inference_bottom_up_pose_model(
            pose_model,
            frame,
            format='xywh',
            dataset='WholeBody'
        )

        for person in pose_results:
            kpts = person['keypoints']  # shape (133, 3)
            # Indeksy dla dłoni w WholeBody: 68-88 lewe (21), 89-109 prawe (21)
            left_hand = kpts[68:68+21]
            right_hand = kpts[89:89+21]
            # Spłaszcz do wektora [x1,y1,z1,...]
            vec_lewe = left_hand.reshape(-1)
            vec_prawe = right_hand.reshape(-1)
            # Połącz obie dłonie
            vec_all = np.concatenate([vec_lewe, vec_prawe])
            cechy.append(vec_all)

    cap.release()
    return cechy


def ekstrahuj_cechy_z_folderu(sciezka_folderu, sciezka_docelowa, pose_model):
    """Ekstrahuje cechy ze wszystkich filmów w folderze i zapisuje je."""
    for plik in os.listdir(sciezka_folderu):
        if plik.lower().endswith(('.mp4', '.avi', '.mkv', '.webm')):
            sciezka_filmu = os.path.join(sciezka_folderu, plik)
            cechy_filmu = ekstrahuj_cechy_z_filmu(sciezka_filmu, pose_model)
            if cechy_filmu:
                nazwa_pliku = f"{os.path.splitext(plik)[0]}.npy"
                sciezka_pliku = os.path.join(sciezka_docelowa, nazwa_pliku)
                np.save(sciezka_pliku, np.array(cechy_filmu))
                logging.info(f"Zapisano cechy z: {sciezka_filmu} do: {sciezka_pliku}")
            else:
                logging.warning(f"Nie udało się wyodrębnić cech z: {sciezka_filmu}")


def ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa, pose_model):
    """Ekstrahuje cechy z wszystkich folderów (słów) w danych i zapisuje je."""
    for slowo in os.listdir(sciezka_zrodlowa):
        sciezka_slowa = os.path.join(sciezka_zrodlowa, slowo)
        if os.path.isdir(sciezka_slowa):
            sciezka_docelowa_slowa = os.path.join(sciezka_docelowa, slowo)
            os.makedirs(sciezka_docelowa_slowa, exist_ok=True)
            ekstrahuj_cechy_z_folderu(sciezka_slowa, sciezka_docelowa_slowa, pose_model)


if __name__ == '__main__':
    # Inicjalizacja modelu MMPose bottom-up WholeBody\ n    
    pose_model = init_pose_model(
        POSE_CONFIG,
        POSE_CHECKPOINT,
        device='cuda:0'  # lub 'cpu'
    )

    # Ścieżki użytkownika\ n    
    sciezka_zrodlowa = 'valid_dataset_skalowane'
    sciezka_docelowa = 'sekwencje_skalowanie_mmpose'

    # Tworzenie folderu docelowego, jeśli nie istnieje
    os.makedirs(sciezka_docelowa, exist_ok=True)

    # Uruchomienie ekstrakcji\ n    
    ekstrahuj_cechy_z_danych(sciezka_zrodlowa, sciezka_docelowa, pose_model)
