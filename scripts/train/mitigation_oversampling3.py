"""
Oversampling (class × skin) + full metrics.
Put this in your project root and `python oversample_logreg_metrics.py`.
Outputs:
  • PNG confusion matrices in reports/
  • CSV of metrics (overall, per-group, gap) in reports/
"""

import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)

# paths
TRAIN_NPZ = "features/bottleneck_train_ft4.npz"
TEST_NPZ  = "features/bottleneck_test_ft4.npz"
TRAIN_CSV = "splits/train_split_4.csv"
TEST_CSV  = "splits/test_split_4.csv"
OUT_DIR   = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

# hyper-knobs
C_REG   = 1.0      # logistic-reg regularisation
MAX_IT  = 2000
SAVE_PLOTS = True
SAVE_CSV   = True

# ---------- build balanced train set ----------
tr = np.load(TRAIN_NPZ, allow_pickle=True)
X  = tr["features"]
meta_tr = pd.read_csv(TRAIN_CSV)
y_str  = meta_tr["emotion_label"].values
skin   = meta_tr["skin_bin_str"].values

le = LabelEncoder(); y = le.fit_transform(y_str)

df = pd.DataFrame(X)
df["y_str"] = y_str
df["y"]     = y
df["skin"]  = skin

# cell counts & target size
cell_counts = df.groupby(["y_str","skin"]).size()
n_max = cell_counts.max()

balanced_parts = []
rng = np.random.default_rng(42)
for (cls, g), n in cell_counts.items():
    subset = df[(df["y_str"]==cls) & (df["skin"]==g)]
    reps   = n_max // n
    rem    = n_max - reps * n
    bal = pd.concat([subset]*reps + [subset.sample(rem, replace=True, random_state=42)],
                    ignore_index=True)
    balanced_parts.append(bal)

balanced = (pd.concat(balanced_parts)
              .sample(frac=1.0, random_state=123)  # shuffle
              .reset_index(drop=True))

X_bal = balanced.iloc[:, :1280].values
y_bal = balanced["y"].values

# ---------- train ----------
clf = LogisticRegression(C=C_REG, max_iter=MAX_IT, multi_class="multinomial")
clf.fit(X_bal, y_bal)

# ---------- test ----------
te = np.load(TEST_NPZ, allow_pickle=True)
Xte = te["features"]
meta_te = pd.read_csv(TEST_CSV)
y_te_str = meta_te["emotion_label"].values
skin_te  = meta_te["skin_bin_str"].values
y_te = le.transform(y_te_str)
y_pred = clf.predict(Xte)

def cm_png(y_true, y_pred, name):
    if not SAVE_PLOTS: return
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(le.classes_)))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45, colorbar=False)
    plt.title(name); plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=160); plt.close()
    print(f"[saved] {path}")

# overall metrics
rows = []
acc = accuracy_score(y_te, y_pred)
f1m = f1_score(y_te, y_pred, average="macro")
f1w = f1_score(y_te, y_pred, average="weighted")
print("\n=== TEST (oversampling class×skin) ===")
print(f"accuracy={acc:.4f} | macro-F1={f1m:.4f} | weighted-F1={f1w:.4f}")
print(classification_report(y_te, y_pred, target_names=le.classes_))
rows.append({"split":"overall","accuracy":acc,"macro_f1":f1m,"weighted_f1":f1w})
cm_png(y_te, y_pred, "cm_overall_oversample")

# subgroup metrics
stats = {}
for g in sorted(pd.unique(skin_te)):
    idx = np.where(skin_te==g)[0]
    acc_g = accuracy_score(y_te[idx], y_pred[idx])
    f1m_g = f1_score(y_te[idx], y_pred[idx], average="macro")
    f1w_g = f1_score(y_te[idx], y_pred[idx], average="weighted")
    print(f"\n--- Group {g} ---")
    print(f"accuracy={acc_g:.4f} | macro-F1={f1m_g:.4f}")
    print(classification_report(y_te[idx], y_pred[idx], target_names=le.classes_))
    stats[g] = (acc_g,f1m_g)
    rows.append({"split":f"group_{g}","accuracy":acc_g,"macro_f1":f1m_g,"weighted_f1":f1w_g})
    cm_png(y_te[idx], y_pred[idx], f"cm_group_{g}_oversample")

# gaps
if {"Light","Dark"}.issubset(stats):
    gap_acc = stats["Light"][0] - stats["Dark"][0]
    gap_f1m = stats["Light"][1] - stats["Dark"][1]
    print("\n=== GAPS (Light − Dark) ===")
    print(f"Δ accuracy = {gap_acc:+.4f}")
    print(f"Δ macro-F1 = {gap_f1m:+.4f}")
    rows.append({"split":"gap_L_minus_D","accuracy":gap_acc,"macro_f1":gap_f1m,"weighted_f1":np.nan})

# CSV summary
if SAVE_CSV:
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"oversample_logreg_metrics.csv"), index=False)
    print("[saved] reports/oversample_logreg_metrics.csv")
