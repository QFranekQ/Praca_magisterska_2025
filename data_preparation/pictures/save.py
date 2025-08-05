import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]
    normalized = [(x - base_x, y - base_y, z - base_z) for (x, y, z) in landmarks]
    flat = np.array(normalized).flatten()
    return flat

DATASET_PATH = "dataset"
output_data = []
output_labels = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue
    
    for file_name in tqdm(os.listdir(label_path), desc=f"Przetwarzanie klasy '{label}'"):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        img_path = os.path.join(label_path, file_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            normalized = normalize_landmarks(landmark_list)
            output_data.append(normalized)
            output_labels.append(label)

df = pd.DataFrame(output_data)
df["label"] = output_labels
df.to_csv("landmark_data.csv", index=False)

print("Zapisano dane do 'landmark_data.csv'")
