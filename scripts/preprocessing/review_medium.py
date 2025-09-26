import pandas as pd
import cv2

# Carica i Medium
df = pd.read_csv("medium_cases.csv")
df["manual_label"] = None

print(f"Loaded {len(df)} medium cases for review.")
print("Premi 'l' per Light, 'd' per Dark, 'q' per uscire.")

for idx, row in df.iterrows():
    img = cv2.imread(row["image_path"])
    if img is None:
        print(f"Image not found: {row['image_path']}")
        continue

    # Ridimensiona se troppo grande (opzionale)
    max_dim = 600
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Aggiungi testo con info
    label_text = f"{idx+1}/{len(df)} | {row['image_name']}"
    cv2.putText(img, label_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostra immagine
    cv2.imshow("Review Medium", img)

    # Attendi input da tastiera
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("l"):
            df.at[idx, "manual_label"] = "Light"
            break
        elif key == ord("d"):
            df.at[idx, "manual_label"] = "Dark"
            break
        elif key == ord("q"):  # uscita rapida
            print("Interrotto dall'utente")
            cv2.destroyAllWindows()
            df.to_csv("medium_cases_labeled.csv", index=False)
            exit()

# Chiudi finestra e salva
cv2.destroyAllWindows()
df.to_csv("medium_cases_labeled.csv", index=False)
print("Saved labeled medium cases to medium_cases_labeled.csv")
