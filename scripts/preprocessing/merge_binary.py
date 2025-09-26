import pandas as pd

# Carica file principali
df = pd.read_csv("skin_tone_annotations.csv")
medium = pd.read_csv("medium_cases_labeled.csv")[["image_name", "manual_label"]]

# Unisci le etichette manuali
df = df.merge(medium, on="image_name", how="left")

# Crea nuova colonna binaria HL
df["skin_tone_HL_binary"] = df.apply(
    lambda x: x["manual_label"] if pd.notna(x["manual_label"]) else (
        x["skin_tone_HL"] if x["skin_tone_HL"] in ["Light", "Dark"] else "unknown"
    ),
    axis=1
)

# Salva nuovo CSV
df.to_csv("skin_tone_annotationsHLbinary.csv", index=False)
print("✅ Saved skin_tone_annotationsHLbinary.csv with Light/Dark labels")
