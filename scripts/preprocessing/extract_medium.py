import pandas as pd

# Carica i CSV
tone_df = pd.read_csv("skin_tone_annotations.csv")
path_df = pd.read_csv("rafdb_emotions.csv")

# Merge per aggiungere i path
df = pd.merge(tone_df, path_df, on="image_name", how="left")

# Estrai i Medium da H–L
medium_df = df[df["skin_tone_HL"] == "Medium"]

# Salva in un nuovo file con i path inclusi
medium_df.to_csv("medium_cases.csv", index=False)

print(f"Extracted {len(medium_df)} Medium cases and saved to medium_cases.csv")
