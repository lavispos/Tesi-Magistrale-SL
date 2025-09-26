import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

tone_fn = "skin_tone_annotationsHLbinary.csv"
path_fn = "rafdb_emotions.csv"
if not os.path.exists(tone_fn) or not os.path.exists(path_fn):
    raise SystemExit("Metti i file skin_tone_annotationsHLbinary.csv e rafdb_emotions.csv nella cartella corrente.")

tone_df = pd.read_csv(tone_fn)
path_df = pd.read_csv(path_fn)
df = pd.merge(tone_df, path_df, on="image_name", how="inner")

# map labels if numeric
raw_vals = sorted(pd.unique(df["emotion_label"].dropna()))
if raw_vals == [0,1,2,3,4,5,6]:
    emap = {0:"surprise",1:"fear",2:"disgust",3:"happiness",4:"sadness",5:"anger",6:"neutral"}
elif raw_vals == [1,2,3,4,5,6,7]:
    emap = {1:"surprise",2:"fear",3:"disgust",4:"happiness",5:"sadness",6:"anger",7:"neutral"}
else:
    emap = None
if emap is not None:
    df["emotion_label"] = df["emotion_label"].map(emap)

# normalize skin tone
def to_light_dark(arr):
    arr = pd.Series(arr)
    if np.issubdtype(arr.dtype, np.number):
        return np.where(arr.astype(int)==1, "Dark", "Light")
    s = arr.astype(str).str.strip().str.lower()
    s = s.replace({"0":"light","1":"dark","l":"light","d":"dark","lt":"light","dk":"dark"})
    return np.where(s.eq("dark"), "Dark", np.where(s.eq("light"), "Light", "Unknown"))

if "skin_tone_HL_binary" in df.columns:
    df["skin_bin_str"] = to_light_dark(df["skin_tone_HL_binary"])
else:
    raise SystemExit("Colonna skin_tone_HL_binary non trovata nel CSV delle annotazioni.")

# prefer manual_label if present
if "manual_label" in df.columns:
    def prefer_manual(r):
        m = r.get("manual_label")
        if pd.notna(m):
            v = str(m).strip().lower()
            if v in ["dark","d"]: return "Dark"
            if v in ["light","l"]: return "Light"
        return r["skin_bin_str"]
    df["skin_bin_str"] = df.apply(prefer_manual, axis=1)

# try to detect official split column (preserve if present)
def detect_subset_column(_df):
    for c in ["subset","split","usage","set","partition"]:
        if c in _df.columns:
            vals = set(str(v).lower() for v in _df[c].dropna().unique())
            if any(v in vals for v in ["train","test","val","validation"]):
                return c
    return None

subset_col = detect_subset_column(df)
train_df = val_df = test_df = None

if subset_col:
    sub = df[subset_col].astype(str).str.lower()
    train_df = df[sub.eq("train")].copy()
    test_df  = df[sub.eq("test")].copy()
    print("Detected official split column:", subset_col)
else:
    # joint stratify on known (no Unknown)
    df_known = df[df["skin_bin_str"] != "Unknown"].copy()
    print("Known samples (Light/Dark):", len(df_known), "Unknown:", (df["skin_bin_str"]=="Unknown").sum())
    if len(df_known) >= 100:
        df_known["joint"] = df_known["emotion_label"].astype(str) + "|" + df_known["skin_bin_str"].astype(str)
        joint_counts = df_known["joint"].value_counts()
        min_count = int(joint_counts.min()) if len(joint_counts)>0 else 0
        print("Joint groups:", len(joint_counts), "min count per joint:", min_count)
        if min_count >= 2:
            train_known, test_known = train_test_split(df_known, test_size=0.2, stratify=df_known["joint"], random_state=SEED)
            train_known, val_known  = train_test_split(train_known, test_size=0.1, stratify=train_known["joint"], random_state=SEED)
            train_df = train_known.reset_index(drop=True)
            val_df   = val_known.reset_index(drop=True)
            test_df  = test_known.reset_index(drop=True)
            print("Used joint stratification on knowns.")
        else:
            print("Joint stratify not possible (groups too small). Falling back to stratify by emotion on full df.")
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["emotion_label"], random_state=SEED)
    else:
        print("Too few knowns; falling back to stratify by emotion on full df.")
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["emotion_label"], random_state=SEED)

# 10% val from train if not already created
if val_df is None:
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["emotion_label"], random_state=SEED)

# save splits
os.makedirs("splits", exist_ok=True)
train_df.to_csv("splits/train_split_4.csv", index=False)
val_df.to_csv("splits/val_split_4.csv", index=False)
test_df.to_csv("splits/test_split_4.csv", index=False)

print("Saved splits to splits/*.csv")
print("Counts in splits (test):")
print(test_df["skin_bin_str"].value_counts(dropna=False).to_string())
