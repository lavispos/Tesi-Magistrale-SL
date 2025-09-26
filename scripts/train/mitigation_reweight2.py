# reweight_logreg_metrics.py — Logistic Regression on bottlenecks with (class×skin) reweighting
import os, numpy as np, pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ========= paths =========
TRAIN_NPZ = "features/bottleneck_train_ft4.npz"
TEST_NPZ  = "features/bottleneck_test_ft4.npz"
TRAIN_CSV = "splits/train_split_4.csv"
TEST_CSV  = "splits/test_split_4.csv"
OUT_DIR   = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= knobs (feel free to tweak) =========
USE_SMOOTH = True        # True -> use 1/sqrt(freq) (stabilizza); False -> 1/freq (più aggressivo)
LOGREG_C   = 1.0         # regolarizzazione (più piccolo => più forte)
MAX_ITER   = 2000
SAVE_PLOTS = True        # salva confusion matrices
SAVE_CSV   = True        # salva tabellina metriche

# ========= load train =========
tr = np.load(TRAIN_NPZ, allow_pickle=True)
Xtr = tr["features"]
meta_tr = pd.read_csv(TRAIN_CSV)
y_tr_str = meta_tr["emotion_label"].values
skin_tr  = meta_tr["skin_bin_str"].values

le = LabelEncoder()
ytr = le.fit_transform(y_tr_str)

# sample weights per (class, skin)
pairs = list(zip(y_tr_str, skin_tr))
cnt = Counter(pairs)
if USE_SMOOTH:
    pair_w = {p: 1.0 / np.sqrt(c) for p, c in cnt.items()}
else:
    pair_w = {p: 1.0 / c for p, c in cnt.items()}
w = np.array([pair_w[(y_tr_str[i], skin_tr[i])] for i in range(len(ytr))], dtype=float)

# ========= train weighted classifier =========
clf = LogisticRegression(C=LOGREG_C, max_iter=MAX_ITER, multi_class='multinomial')
clf.fit(Xtr, ytr, sample_weight=w)

# ========= load test =========
te = np.load(TEST_NPZ, allow_pickle=True)
Xte = te["features"]
meta_te = pd.read_csv(TEST_CSV)
y_te_str = meta_te["emotion_label"].values
skin_te  = meta_te["skin_bin_str"].values
yte = le.transform(y_te_str)

yp = clf.predict(Xte)

# ========= helpers =========
def print_overall(y_true, y_pred, title="OVERALL"):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {title} ===")
    print(f"accuracy={acc:.4f} | macro-F1={f1m:.4f} | weighted-F1={f1w:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return {"split": title, "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w}

def per_group_metrics(y_true, y_pred, groups, label):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    f1m = f1_score(y_true, y_pred, average="macro")
    pr, rc, f1c, sup = precision_recall_fscore_support(y_true, y_pred, labels=np.unique(y_true), zero_division=0)
    print(f"\n--- Group: {label} ---")
    print(f"accuracy={acc:.4f} | macro-F1={f1m:.4f} | weighted-F1={f1w:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return {"split": f"group_{label}", "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w}

def save_cm(y_true, y_pred, name):
    if not SAVE_PLOTS: return
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(le.classes_)))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45, colorbar=False)
    plt.title(name)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"cm_{name}.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[saved] {path}")

# ========= overall =========
rows = []
rows.append(print_overall(yte, yp, "TEST (weighted reweighting)"))
save_cm(yte, yp, "overall_reweight")

# ========= per-group (Light/Dark) =========
rows_g = {}
for g in sorted(pd.unique(skin_te)):
    idx = np.where(skin_te == g)[0]
    rows_g[g] = per_group_metrics(yte[idx], yp[idx], skin_te[idx], g)
    save_cm(yte[idx], yp[idx], f"group_{g}_reweight")

# ========= gaps =========
if {"Light","Dark"}.issubset(set(rows_g.keys())):
    gap_acc = rows_g["Light"]["accuracy"] - rows_g["Dark"]["accuracy"]
    gap_f1m = rows_g["Light"]["macro_f1"] - rows_g["Dark"]["macro_f1"]
    print(f"\n=== GAPS (Light - Dark) ===")
    print(f"Δ accuracy = {gap_acc:+.4f}")
    print(f"Δ macro-F1 = {gap_f1m:+.4f}")
    rows.append({"split": "gap_L_minus_D", "accuracy": gap_acc, "macro_f1": gap_f1m, "weighted_f1": np.nan})

# ========= dump CSV summary =========
if SAVE_CSV:
    df = pd.DataFrame([*rows, *rows_g.values()])
    out_csv = os.path.join(OUT_DIR, "reweight_logreg_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

# ========= tiny audit of weights =========
w_ser = pd.Series(w, name="sample_weight")
print("\n[weights] mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(w_ser.mean(), w_ser.std(), w_ser.min(), w_ser.max()))
top_pairs = Counter(pairs).most_common(3)
bot_pairs = sorted(Counter(pairs).items(), key=lambda kv: kv[1])[:3]
print("[weights] most common pairs:", top_pairs)
print("[weights] least  common pairs:", bot_pairs)
