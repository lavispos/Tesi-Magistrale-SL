import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import face_recognition
from skimage import color
import time
import matplotlib.pyplot as plt

start_time = time.time()

# === 1. Carica CSV immagini ===
df = pd.read_csv("rafdb_emotions.csv")
# === 2. Risultati
results = []
# === 3. Loop su tutte le immagini ===
for _, row in tqdm(df.iterrows(), total=len(df), desc="Elaborazione immagini"):
    image_name = row["image_name"]
    path = row["image_path"]
    try:
        # --- Carica immagine ---
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Immagine non caricata")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # --- Check grayscale / low-color (intera immagine) ---
        channel_var = np.var(img_rgb, axis=2).mean()
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        s_mean_global = img_hsv[:,:,1].mean()

        if channel_var < 5 or s_mean_global < 20:
            raise ValueError("Immagine grayscale o low-color")
        # --- Rilevamento volto ---
        faces = face_recognition.face_locations(img_rgb)
        if not faces:
            raise ValueError("Nessun volto trovato")
        top, right, bottom, left = faces[0]
        face_crop = img_rgb[top:bottom, left:right]
        # --- Segmentazione pelle ---
        face_ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_RGB2YCrCb)
        skin_mask = cv2.inRange(face_ycrcb, (0, 133, 77), (255, 173, 127))
        skin = cv2.bitwise_and(face_crop, face_crop, mask=skin_mask)
        # --- Check face-paint / colori innaturali (sui pixel pelle) ---
        skin_hsv = cv2.cvtColor(skin, cv2.COLOR_RGB2HSV)
        s_mean_skin = skin_hsv[:,:,1][skin_mask > 0].mean()
        h_mean_skin = skin_hsv[:,:,0][skin_mask > 0].mean()
        # Hue in OpenCV è [0,179] → circa 0–25 = toni naturali della pelle
        if s_mean_skin > 150 or not (0 <= h_mean_skin <= 25):
            raise ValueError("Immagine con face paint / colori non naturali")
        # --- Conversione in Lab ---
        skin_lab = color.rgb2lab(skin / 255.0)
        skin_pixels = skin_lab[skin_mask > 0]
        if skin_pixels.size == 0:
            raise ValueError("Nessun pixel pelle valido")
        # --- Calcolo metriche ---
        L_mean = np.mean(skin_pixels[:, 0])
        a_mean = np.mean(skin_pixels[:, 1])
        b_mean = np.mean(skin_pixels[:, 2])
        hue = np.degrees(np.arctan2(b_mean, a_mean))

        # --- Calcolo ITA ---
        ita = np.degrees(np.arctan2(L_mean - 50, b_mean))
        # --- Classificazione ITA ---
        if ita > 55:
            tone_ita = "Light"
        elif ita >= 30:
            tone_ita = "Medium"
        else:
            tone_ita = "Dark"
        # --- Classificazione HL ---
        if L_mean > 67:
            tone = "Light"
        elif L_mean >= 37:
            tone = "Medium"
        else:
            tone = "Dark"

        # --- Risultato ---
        results.append({
            "image_name": image_name,
            "L*": round(L_mean, 2),
            "a*": round(a_mean, 2),
            "b*": round(b_mean, 2),
            "Hue": round(hue, 2),
            "ITA": round(ita, 2),
            "skin_tone_HL": tone,    
            "skin_tone_ITA": tone_ita,
            "override_applied": False
        })
    except Exception as e:
        results.append({
            "image_name": image_name,
            "L*": None,
            "a*": None,
            "b*": None,
            "Hue": None,
            "skin_tone_HL": "unknown",
            "skin_tone_ITA": "unknown",
            "override_applied": False
        })
    # Salvataggio parziale ogni 10 immagini
    if len(results) % 10 == 0:
        pd.DataFrame(results).to_csv("skin_tone_annotations_temp.csv", index=False)

# === 4. Salva in CSV finale ===
result_df = pd.DataFrame(results)
result_df.to_csv("skin_tone_annotations.csv", index=False)
print("\nSkin tone salvati in 'skin_tone_annotations.csv'")
elapsed = time.time() - start_time
print(f"\n Tempo totale: {elapsed:.2f} secondi")
error_count = sum(1 for r in results if r["skin_tone_HL"] == "unknown")
print(f"\n Immagini non processate correttamente: {error_count}")
