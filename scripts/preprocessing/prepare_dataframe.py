import os
import pandas as pd

# === 1. Percorso alle immagini
image_dir = os.path.join("Image", "aligned", "aligned")
label_file = "list_patition_label.txt"
# === 2. Carica le etichette da file
df = pd.read_csv(label_file, sep=" ", header=None, names=["image_name", "emotion_label"])
# === 3. Pulizia e correzione nomi
df["image_name"] = df["image_name"].str.strip().str.replace(".jpg", "_aligned.jpg")
df["emotion_label"] = df["emotion_label"].astype(int)
# === 4. Costruisce il percorso assoluto alle immagini
df["image_path"] = df["image_name"].apply(lambda x: os.path.join(image_dir, x))
# === 5. Verifica che i file esistano
df["exists"] = df["image_path"].apply(os.path.exists)
found = df["exists"].sum()
print(f"Immagini trovate: {found} su {len(df)}")
# === 6. Filtra solo quelle esistenti
df = df[df["exists"]].drop(columns=["exists"])
# === 7. Salva il file CSV finale
df.to_csv("rafdb_emotions.csv", index=False)
print("Salvato: rafdb_emotions.csv")
