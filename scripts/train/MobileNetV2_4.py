# MobileNetV2_4.py
# MobileNetV2 baseline (RAF-DB) with robust joint-stratified split (emotion x skin) and outputs *_4

import os, sys, json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==============
# 0) Config
# ==============
SEED = 42
IMG_SIZE = (224, 224)
BATCH = 32
HEAD_EPOCHS = 15
FT_EPOCHS = 20
UNFREEZE_LAST = 20
USE_PREPROCESS_INPUT = True  # True = MobileNetV2 preprocess_input ([-1,1]); False = rescale 1/255

os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("splits", exist_ok=True)

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# helper filenames with _4 suffix
CKPT_HEAD = "mobilenet_head_4.weights.h5"
CKPT_FT   = "mobilenet_ft_4.weights.h5"
CSV_HEAD  = "logs/history_head_4.csv"
CSV_FT    = "logs/history_ft_4.csv"
REPORT_CLASS = "reports/classification_report_4.txt"
REPORT_FAIR  = "reports/fairness_by_skin_4.txt"

# ==============
# 1) Load CSVs
# ==============
tone_fn = "skin_tone_annotationsHLbinary.csv"
path_fn = "rafdb_emotions.csv"
if not os.path.exists(tone_fn) or not os.path.exists(path_fn):
    raise FileNotFoundError("Metti i file skin_tone_annotationsHLbinary.csv e rafdb_emotions.csv nella cartella corrente.")

tone_df = pd.read_csv(tone_fn)
path_df = pd.read_csv(path_fn)

# Merge on image_name
if "image_name" not in tone_df.columns or "image_name" not in path_df.columns:
    raise ValueError("Assicurati che entrambi i CSV contengano la colonna 'image_name' per il merge.")
df = pd.merge(tone_df, path_df, on="image_name", how="inner")
print("TOTAL rows merged:", len(df))

# ==============
# 2) Label mapping (robusto a 0–6 o 1–7)
# ==============
raw_vals = sorted(pd.unique(df["emotion_label"].dropna()))
if raw_vals == [0,1,2,3,4,5,6]:
    EMAP = {0:"surprise",1:"fear",2:"disgust",3:"happiness",4:"sadness",5:"anger",6:"neutral"}
elif raw_vals == [1,2,3,4,5,6,7]:
    EMAP = {1:"surprise",2:"fear",3:"disgust",4:"happiness",5:"sadness",6:"anger",7:"neutral"}
else:
    EMAP = None

if EMAP is not None:
    df["emotion_label"] = df["emotion_label"].map(EMAP)

# cleanup
needed_cols = ["image_path", "emotion_label"]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Mancano colonne nel CSV unito: {missing}")
df = df.dropna(subset=["emotion_label", "image_path"])

# ==============
# 3) Skin tone normalization (Light/Dark/Unknown)
# ==============
def to_light_dark(arr):
    arr = pd.Series(arr)
    if np.issubdtype(arr.dtype, np.number):
        return np.where(arr.astype(int)==1, "Dark", "Light")
    s = arr.astype(str).str.strip().str.lower()
    s = s.replace({"0":"light","1":"dark"})
    s = s.replace({"l":"light","d":"dark"})
    s = s.replace({"lt":"light","dk":"dark"})
    return np.where(s.eq("dark"), "Dark", np.where(s.eq("light"), "Light", "Unknown"))

if "skin_tone_HL_binary" in df.columns:
    df["skin_bin_str"] = to_light_dark(df["skin_tone_HL_binary"])
else:
    cand = [c for c in df.columns if "skin" in c.lower() and "tone" in c.lower()]
    raise ValueError(f"Colonna skin tone non trovata. Nomi disponibili: {df.columns.tolist()}")

# if manual_label exists, prefer it (useful if you manually corrected mediums)
if "manual_label" in df.columns:
    def prefer_manual(r):
        m = r.get("manual_label")
        if pd.notna(m):
            v = str(m).strip().lower()
            if v in ["dark","d"]: return "Dark"
            if v in ["light","l"]: return "Light"
        return r["skin_bin_str"]
    df["skin_bin_str"] = df.apply(prefer_manual, axis=1)

print("\nCounts overall (skin_bin_str):")
print(df["skin_bin_str"].value_counts(dropna=False).to_string())

# ==============
# 4) Robust split: joint stratify on known (emotion x skin) if possible
# ==============
def detect_subset_column(_df):
    for c in ["subset", "split", "usage", "set", "partition"]:
        if c in _df.columns:
            vals = set(str(v).lower() for v in _df[c].dropna().unique())
            if any(v in vals for v in ["train","test","val","validation"]):
                return c
    return None

# try to detect official split first (preserve if present)
subset_col = detect_subset_column(df)

train_df = None
val_df = None
test_df = None

if subset_col:
    sub = df[subset_col].astype(str).str.lower()
    train_df = df[sub.eq("train")].copy()
    test_df  = df[sub.eq("test")].copy()
    print("Detected official split column:", subset_col)
else:
    # attempt joint stratify on knowns
    df_known = df[df["skin_bin_str"] != "Unknown"].copy()
    print(f"Known samples (Light/Dark): {len(df_known)} | Unknown: {(df['skin_bin_str']=='Unknown').sum()}")
    if len(df_known) < 100:
        # too few knowns for reliable joint stratify -> fallback
        print("Too few known samples for joint stratify; falling back to emotion stratify.")
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["emotion_label"], random_state=SEED)
        train_df = train_df.copy(); test_df = test_df.copy()
    else:
        df_known["joint"] = df_known["emotion_label"].astype(str) + "|" + df_known["skin_bin_str"].astype(str)
        joint_counts = df_known["joint"].value_counts()
        min_count = int(joint_counts.min())
        print("Joint groups:", len(joint_counts), "min count per joint:", min_count)
        if min_count >= 2:
            # do joint stratify on knowns
            train_known, test_known = train_test_split(df_known, test_size=0.2, stratify=df_known["joint"], random_state=SEED)
            train_known, val_known = train_test_split(train_known, test_size=0.1, stratify=train_known["joint"], random_state=SEED)
            train_df = train_known.reset_index(drop=True)
            val_df   = val_known.reset_index(drop=True)
            test_df  = test_known.reset_index(drop=True)
            print("Used joint stratification (emotion x skin) on known samples.")
        else:
            # fallback to stratify by emotion on full df
            print("Joint stratification not possible (some joint groups too small). Falling back to stratify by emotion on full dataset.")
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["emotion_label"], random_state=SEED)
            train_df = train_df.copy(); test_df = test_df.copy()

# 10% validation dal train (stratificata per emozione or joint if available)
if val_df is None:
    # if we fell back and train_df came from full df, stratify by emotion
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["emotion_label"], random_state=SEED)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# Save canonical splits
train_df.to_csv("splits/train_split_4.csv", index=False)
val_df.to_csv("splits/val_split_4.csv", index=False)
test_df.to_csv("splits/test_split_4.csv", index=False)
print("\nSaved splits in splits/: train_split_4.csv / val_split_4.csv / test_split_4.csv")

print("\nCounts in splits (final):")
print("TRAIN:", train_df["skin_bin_str"].value_counts(dropna=False).to_string())
print("VAL:  ", val_df["skin_bin_str"].value_counts(dropna=False).to_string())
print("TEST: ", test_df["skin_bin_str"].value_counts(dropna=False).to_string())

# ==============
# 5) Generators
# ==============
if USE_PREPROCESS_INPUT:
    datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input)
    datagen_eval  = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    datagen_train = ImageDataGenerator(rescale=1./255)
    datagen_eval  = ImageDataGenerator(rescale=1./255)

def make_gen(_df, shuffle, datagen):
    return datagen.flow_from_dataframe(
        _df,
        x_col="image_path",
        y_col="emotion_label",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH,
        shuffle=shuffle,
        seed=SEED
    )

train_gen = make_gen(train_df, shuffle=True,  datagen=datagen_train)
val_gen   = make_gen(val_df,   shuffle=False, datagen=datagen_eval)
test_gen  = make_gen(test_df,  shuffle=False, datagen=datagen_eval)

# ==============
# 6) Model
# ==============
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(train_gen.class_indices), activation="softmax")
])

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# ==============
# 7) Training – Stage 1 (head-only)
# ==============
early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
sched = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
ckpt_head = ModelCheckpoint(CKPT_HEAD, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
csv_head  = CSVLogger(CSV_HEAD, append=False)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=HEAD_EPOCHS,
    callbacks=[early, sched, ckpt_head, csv_head],
    verbose=1
)

# ==============
# 8) Fine-tuning
# ==============
model.load_weights(CKPT_HEAD)

base_model = model.layers[0]
base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAST]:
    layer.trainable = False
for layer in base_model.layers[-UNFREEZE_LAST:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary(line_length=120)

early_ft = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
sched_ft = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
# ckpt_ft  = ModelCheckpoint(CKPT_FT, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
# callback separati per Stage-2 (consigliato)
# SALVA L'INTERO MODELLO (.keras) per essere sicuri di avere un file caricabile intero
ckpt_ft = ModelCheckpoint("mobilenet_ft_4.keras",
                          monitor="val_loss",
                          save_best_only=True,
                          save_weights_only=False,   # salva l'intero modello .keras
                          verbose=1)
# tieni anche un csv logger se vuoi
csv_ft = CSVLogger("logs/history_ft.csv", append=False)
# csv_ft   = CSVLogger(CSV_FT, append=False)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FT_EPOCHS,
    callbacks=[early_ft, sched_ft, ckpt_ft, csv_ft],
    verbose=1
)

# 1) salva l'intero modello (formato .keras)
try:
    model.save("mobilenet_ft_4_full.keras")   # file caricabile con tf.keras.models.load_model
    print("[SAVED] mobilenet_ft_4_full.keras")
except Exception as e:
    print("[WARN] Non ho potuto salvare full model .keras:", e)

# 2) salva i pesi in formato HDF5 compatibile (solo pesi)
try:
    model.save_weights("mobilenet_ft_4_fullcompat.h5")
    print("[SAVED] mobilenet_ft_4_fullcompat.h5 (weights only)")
except Exception as e:
    print("[WARN] Non ho potuto salvare weights .h5:", e)

# ==============
# 9) Final evaluation on TEST (load best weights)  -- robust version
# ==============

# Caricamento modello o pesi migliori
full_candidates = [
    "mobilenet_ft_4_full.keras",
    "mobilenet_ft_4.keras",
    "mobilenet_ft_4_full.h5"
]
loaded_full = False
for fp in full_candidates:
    if os.path.exists(fp):
        try:
            print(f"[INFO] Carico modello/pesi da: {fp}")
            model = tf.keras.models.load_model(fp)
            print(f"[OK] Full model caricato da {fp}")
            loaded_full = True
            break
        except Exception as e:
            print(f"[WARN] Impossibile caricare full model {fp}: {e}. Provo prossimo candidato.")

if not loaded_full:
    if os.path.exists(CKPT_FT):
        print(f"[INFO] Carico pesi da: {CKPT_FT}")
        model.load_weights(CKPT_FT)
    elif os.path.exists(CKPT_HEAD):
        print(f"[INFO] Carico pesi da (head): {CKPT_HEAD}")
        model.load_weights(CKPT_HEAD)
    else:
        print("[WARN] Nessun file di pesi trovato; usiamo modello così com'è (ImageNet backbone).")

# Valutazione su test
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"[TEST] loss={loss:.4f} | acc={acc:.4f}")

# Predizioni e costruzione y_true e y_pred

# Provo a prendere y_true_idx dal generator
y_true_idx = getattr(test_gen, "classes", None)

# Predizioni
probs = model.predict(test_gen, verbose=0)
y_pred_idx = probs.argmax(axis=1)

# Inversione da indice a label coerente
idx_to_label = {v:k for k,v in train_gen.class_indices.items()}

# Ricostruisco y_true
if y_true_idx is None:
    try:
        y_true_idx = test_df["emotion_label"].map({v:k for k,v in train_gen.class_indices.items()}).values
    except Exception:
        y_true_idx = getattr(test_gen, "labels", None)

if y_true_idx is not None and np.issubdtype(np.array(y_true_idx).dtype, np.integer):
    y_true = np.array([idx_to_label[i] for i in y_true_idx])
else:
    if "emotion_label" in test_df.columns:
        y_true = test_df["emotion_label"].values
    else:
        y_true = np.array(["unknown"] * len(y_pred_idx))

# y_pred in label testuali
y_pred = np.array([idx_to_label[i] for i in y_pred_idx])

# Scrivo report globale
report_txt = classification_report(y_true, y_pred, digits=4, zero_division=0)
print(report_txt)
with open(REPORT_CLASS, "w") as f:
    f.write(f"[TEST] loss={loss:.4f} | acc={acc:.4f}\n\n")
    f.write(report_txt)

# Salvo predizioni per analisi successive
out_df = pd.DataFrame({
    "image_path": test_df["image_path"].values,
    "skin_tone": test_df.get("skin_bin_str", pd.Series(["Unknown"]*len(test_df))).values,
    "y_true": y_true,
    "y_pred": y_pred
})
out_df.to_csv("reports/test_predictions_4.csv", index=False)

# ==============
# 10) Fairness Light vs Dark - solo con classi Light/Dark conosciute
# ==============
skin_test = test_df["skin_bin_str"].values

def group_report(mask, name):
    if mask.sum() == 0:
        return f"{name}: 0 samples\n"
    rep = classification_report(y_true[mask], y_pred[mask], digits=4, zero_division=0)
    header = f"\n== {name} group ==\nSamples: {int(mask.sum())}\n"
    print(header + rep)
    return header + rep

mask_light = (skin_test == "Light")
mask_dark  = (skin_test == "Dark")

text = "[FAIRNESS BREAKDOWN]\n"
text += group_report(mask_light, "Light")
text += group_report(mask_dark,  "Dark")
with open(REPORT_FAIR, "w") as f:
    f.write(text)

# Salvo predizioni (ridondante ma comodo)
out_df.to_csv("reports/test_predictions_4.csv", index=False)
print("Done. Reports in ./reports, logs in ./logs, splits in ./splits")

# ==============
# 11) Confusion matrices (overall + Light/Dark)
# ==============
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ordered_labels = [idx_to_label[i] for i in range(len(train_gen.class_indices))]

def plot_cm(cm, labels, title, out_path, normalize=False, figsize=(7,6), dpi=300):
    cm_show = cm.astype(float)
    if normalize:
        row = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, row, out=np.zeros_like(cm_show), where=row>0)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_show, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{cm_show[i,j]:.1%}" if normalize else f"{int(cm[i,j])}"
            ax.text(j, i, txt, ha="center", va="center")
    fig.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

cm_all = confusion_matrix(y_true, y_pred, labels=ordered_labels)
plot_cm(cm_all, ordered_labels, "Confusion matrix (overall, counts)",       "reports/cm_overall_counts_4.png", normalize=False)
plot_cm(cm_all, ordered_labels, "Confusion matrix (overall, row-normalized)","reports/cm_overall_norm_4.png",   normalize=True)

cm_L = confusion_matrix(y_true[mask_light], y_pred[mask_light], labels=ordered_labels)
cm_D = confusion_matrix(y_true[mask_dark],  y_pred[mask_dark],  labels=ordered_labels)

plot_cm(cm_L, ordered_labels, "Confusion matrix (Light, counts)",        "reports/cm_light_counts_4.png", normalize=False)
plot_cm(cm_L, ordered_labels, "Confusion matrix (Light, row-normalized)","reports/cm_light_norm_4.png",   normalize=True)
plot_cm(cm_D, ordered_labels, "Confusion matrix (Dark, counts)",         "reports/cm_dark_counts_4.png",  normalize=False)
plot_cm(cm_D, ordered_labels, "Confusion matrix (Dark, row-normalized)", "reports/cm_dark_norm_4.png",    normalize=True)

print("CM salvate in reports/: cm_overall_*.png, cm_light_*.png, cm_dark_*.png")
