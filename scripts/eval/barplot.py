import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Carica i CSV ===
tone_df = pd.read_csv("skin_tone_annotationsHLbinary.csv")
path_df = pd.read_csv("rafdb_emotions.csv")
# === 2. Merge per aggiungere le emozioni ===
df = pd.merge(tone_df, path_df, on="image_name", how="left")
print("Colonne disponibili:", df.columns)

# === 3. griglia esempi
import random
import cv2
import matplotlib.pyplot as plt
# Prendi 5 esempi Light e 5 Dark
light_examples = df[df["skin_tone_HL_binary"] == "Light"].sample(5, random_state=42)
dark_examples = df[df["skin_tone_HL_binary"] == "Dark"].sample(5, random_state=42)
examples = pd.concat([light_examples, dark_examples])
plt.figure(figsize=(15, 6))
for i, row in enumerate(examples.itertuples(), 1):
    img = cv2.imread(row.image_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 5, i)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"{row.skin_tone_HL_binary}", fontsize=10)
plt.suptitle("Light and Dark examples", fontsize=16)
plt.tight_layout()
plt.show()


# === 4. Barplot Light vs Dark ===
# Dizionario di mapping (RAF-DB 7 basic emotions)
emotion_map = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral"
}
# Applica la mappatura
df["emotion_name"] = df["emotion_label"].map(emotion_map)
# Usa emotion_name nel barplot
method = "skin_tone_HL_binary"
custom_palette = {
    "Light": "#f5cba7",  # beige chiaro
    "Dark": "#5d3a1a"    # marrone scuro
}
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df[df[method] != "unknown"],
    x="emotion_name",
    hue=method,
    order=df["emotion_name"].value_counts().index,
    palette=custom_palette
)
plt.title(f"Distribution of emotions by skin tone")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.legend(title="Skin tone")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 5. counts
import pandas as pd
# Totale per skin tone
totals = df["skin_tone_HL_binary"].value_counts()
print("Totale per skin tone:")
print(totals)
print()
# Distribuzione per emozione × skin tone
dist = pd.crosstab(df["emotion_name"], df["skin_tone_HL_binary"])
print("Distribuzione per emozione:")
print(dist)
