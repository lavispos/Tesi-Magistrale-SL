import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Cartella dove salvare le immagini escluse
excluded_dir = "excluded"
os.makedirs(excluded_dir, exist_ok=True)

# Carica il CSV originale con le immagini
df = pd.read_csv("rafdb_emotions.csv")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Check immagini"):
    image_name = row["image_name"]
    path = row["image_path"]
    try:
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # === Controllo grayscale / low-color ===
        channel_diff = (
            np.abs(img_rgb[:,:,0].astype(int) - img_rgb[:,:,1].astype(int)).mean() +
            np.abs(img_rgb[:,:,1].astype(int) - img_rgb[:,:,2].astype(int)).mean() +
            np.abs(img_rgb[:,:,0].astype(int) - img_rgb[:,:,2].astype(int)).mean()
        )
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        s_mean_global = img_hsv[:,:,1].mean()
        if channel_diff < 10 or s_mean_global < 20:
            reason = "grayscale_lowcolor"
            out_path = os.path.join(excluded_dir, f"{reason}_{image_name}")
            cv2.imwrite(out_path, img)
            continue    
    except Exception as e:
        # Se qualcosa va storto, la salvo lo stesso per analizzarla
        reason = "error"
        out_path = os.path.join(excluded_dir, f"{reason}_{image_name}")
        cv2.imwrite(out_path, img)

# === Salva un collage di esempi "grayscale/lowcolor" ===
import glob, os
import cv2
import matplotlib.pyplot as plt
excluded_dir = "excluded"              
out_fig = "docs/figures/grayscale_examples.png"  # dove salvare la figura 

# prendi fino a 15 esempi (3x5) tra quelli marcati come grayscale/lowcolor
paths = sorted(glob.glob(os.path.join(excluded_dir, "grayscale_lowcolor_*")))[:15]
if len(paths) == 0:
    print("Nessuna immagine 'grayscale_lowcolor_' trovata in", excluded_dir)
else:
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    rows, cols = 3, 5
    take = min(len(paths), rows*cols)

    plt.figure(figsize=(15, 9))
    for i, p in enumerate(paths[:take], start=1):
        img = cv2.imread(p)
        if img is None: 
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(rows, cols, i)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(os.path.basename(p)[:28], fontsize=8)  # titolo corto
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f" Salvata figura collage: {out_fig}")
