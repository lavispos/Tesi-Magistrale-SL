import pandas as pd
import matplotlib.pyplot as plt
import cv2

# === 1. Carica i CSV
tone_df = pd.read_csv("skin_tone_annotations.csv")
path_df = pd.read_csv("rafdb_emotions.csv")
# === 2. Merge per ottenere i path completi
df = pd.merge(tone_df, path_df, on="image_name", how="left")
df = df[(df["skin_tone_HL"] != "unknown") & (df["skin_tone_ITA"] != "unknown")]
# crea le griglie per gli esempi
def show_samples_grid(df, col, method_name):
    groups = ["Light", "Medium", "Dark"]
    fig, axes = plt.subplots(len(groups), 5, figsize=(15, 9))  # 3 righe × 5 colonne
    for i, group in enumerate(groups):
        # subset = df[df[col] == group].sample(n=5)
        subset = df[df[col] == group]
        take = min(5, len(subset))
        subset = subset.sample(n=take, random_state=42)
        for j, row in enumerate(subset.itertuples()):
            img = cv2.imread(row.image_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img_rgb)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"{group}", fontsize=9)
    plt.suptitle(f"Examples classified with {method_name}", fontsize=16)
    plt.tight_layout()
    plt.show()
# === 3. Visualizza per entrambi i metodi
show_samples_grid(df, "skin_tone_HL", "H–L")
show_samples_grid(df, "skin_tone_ITA", "ITA")


# === 4. Fitzpatrick mapping indicativo (ITA in gradi)
# Tipo I > 55, Tipo II 41-55, Tipo III 28-41, Tipo IV 10-28, Tipo V -30-10, Tipo VI < -30
import seaborn as sns
import matplotlib.pyplot as plt
# Filtra valori ITA validi
ita_values = df["ITA"].dropna()
plt.figure(figsize=(12, 6))
# Istogramma + density
sns.histplot(ita_values, kde=True, bins=50, color="skyblue", edgecolor=None, stat="density")
# Range Fitzpatrick (approssimati dal paper)
ranges = [
    (55, 100, "FST I"),
    (41, 55, "FST II"),
    (28, 41, "FST III"),
    (10, 28, "FST IV"),
    (-30, 10, "FST V"),
    (-100, -30, "FST VI"),
]
#fst_colors = [
#    "#ffe0bd",  # I
#    "#ffcd94",  # II
#    "#eac086",  # III
#   "#d1a17a",  # IV
#    "#a1665e",  # V
#    "#503335",  # VI
#] # non mi convincono i colori
fst_colors = [
    "#fff5e1",  # FST I (very fair)
    "#fdd9b5",  # FST II (fair)
    "#fcb97d",  # FST III (light medium)
    "#e59866",  # FST IV (medium)
    "#a5694f",  # FST V (dark)
    "#5d3a1a",  # FST VI (very dark)
]
# sfondo colorato per i range Fitzpatrick
for (low, high, label), color in zip(ranges, fst_colors):
    plt.axvspan(low, high, alpha=0.3, color=color, label=label)
# Linee verticali per Light/Medium/Dark thresholds (TrustSkin)
plt.axvline(55, color="black", linestyle="--", label="Light threshold (55°)")
plt.axvline(30, color="black", linestyle="--", label="Medium threshold (30°)")
plt.xlabel("ITA value (degrees)")
plt.ylabel("Density")
plt.title("Distribution of ITA with Fitzpatrick scale")
plt.legend()
plt.tight_layout()
plt.show()


# === 5. barplot per emozioni/tono
#import seaborn as sns
#import matplotlib.pyplot as plt
#print(df.columns)
# Usa il dataframe mergeato (df) che ha sia emozioni che skin_tone
# Scegli se usare HL o ITA
#method = "skin_tone_HL"   # oppure "skin_tone_ITA"
# Crea il barplot
#plt.figure(figsize=(10, 6))
#sns.countplot(
#   data=df[df[method] != "unknown"],
#    x="emotion_label",          # colonna delle emozioni (assicurati sia questo il nome)
#    hue=method,           # gruppi Light/Medium/Dark
#    order=df["emotion_label"].value_counts().index  # ordina per frequenza
#)
#plt.title(f"Distribution of emotions by skin tone ({method})")
#plt.xlabel("Emotion")
#plt.ylabel("Count")
#plt.legend(title="Skin tone")
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()
