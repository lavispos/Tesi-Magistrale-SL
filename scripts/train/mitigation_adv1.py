#!/usr/bin/env python3
# MobileNetV2_adv_inprocess.py
# End-to-end adversarial debiasing with two phases:
#   (1) head-only training (backbone frozen)
#   (2) fine-tuning unfreezing last N layers of the backbone
#
# Usage (example):
#   python MobileNetV2_adv_inprocess.py \
#       --splits_dir splits \
#       --img_root . \
#       --batch 32 \
#       --img_size 224 224 \
#       --alpha 0.5 \
#       --lambda_grl 1.0 \
#       --freeze_epochs 3 \
#       --ft_epochs 15 \
#       --unfreeze_last 20 \
#       --lr 1e-4

import os, argparse, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import math

os.makedirs("reports", exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--img_root", default=".")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--img_size", type=int, nargs=2, default=(224,224))
    p.add_argument("--alpha", type=float, default=0.5, help="weight on skin loss")
    p.add_argument("--lambda_grl", type=float, default=1.0, help="GRL strength")
    p.add_argument("--freeze_epochs", type=int, default=3)
    p.add_argument("--ft_epochs", type=int, default=15)
    p.add_argument("--unfreeze_last", type=int, default=20, help="how many last layers of backbone to unfreeze")
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()

# ---------- data ----------
def load_split_csv(path):
    df = pd.read_csv(path)
    # keep only Light/Dark rows
    df = df[df["skin_bin_str"].isin(["Light","Dark"])].reset_index(drop=True)
    return df

def make_label_encoders(df_train):
    le_y = LabelEncoder().fit(df_train["emotion_label"])
    le_g = LabelEncoder().fit(df_train["skin_bin_str"])
    return le_y, le_g

def tf_load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # MobileNetV2 preprocessing ([-1,1])
    return img

def make_ds(df, img_root, le_y, le_g, img_size, batch, shuffle=False):
    paths = tf.constant([os.path.join(img_root, p) for p in df["image_path"].tolist()])
    y_int = tf.constant(le_y.transform(df["emotion_label"]))
    g_int = tf.constant(le_g.transform(df["skin_bin_str"]))
    num_classes = len(le_y.classes_)
    num_groups  = len(le_g.classes_)
    def gen(i):
        x = tf_load_image(paths[i], img_size)
        y = tf.one_hot(y_int[i], num_classes)
        g = tf.one_hot(g_int[i], num_groups)
        return x, {"emotion": y, "skin": g}
    ds = tf.data.Dataset.from_tensor_slices(tf.range(len(paths)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    ds = ds.map(gen, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)
    return ds


def make_ds_balanced(df, img_root, le_y, le_g, img_size, batch, shuffle=False):
    df = df.copy()
    gvals = le_g.transform(df["skin_bin_str"])
    dsets = []
    for g in np.unique(gvals):
        dfg = df[gvals==g]
        ds_ids = tf.data.Dataset.from_tensor_slices(tf.range(len(dfg)))
        if shuffle: ds_ids = ds_ids.shuffle(buffer_size=len(dfg), reshuffle_each_iteration=True)
        paths = [os.path.join(img_root, p) for p in dfg["image_path"].tolist()]
        y_int = le_y.transform(dfg["emotion_label"])
        g_int = le_g.transform(dfg["skin_bin_str"])
        paths = tf.constant(paths); y_int = tf.constant(y_int); g_int = tf.constant(g_int)
        def gen(i):
            x = tf_load_image(paths[i], img_size)
            y = tf.one_hot(y_int[i], len(le_y.classes_))
            g = tf.one_hot(g_int[i], len(le_g.classes_))
            return x, {"emotion": y, "skin": g}
        dsg = ds_ids.map(gen, num_parallel_calls=AUTOTUNE)
        dsets.append(dsg.repeat())  # per campionamento continuo
    ds = tf.data.Dataset.sample_from_datasets(dsets, weights=[1/len(dsets)]*len(dsets))
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds


# ---------- Gradient Reversal Layer (fix: single-input custom_gradient) ----------
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

class FairSelect(callbacks.Callback):
    def __init__(self, ds_val, w_gap=0.4):
        super().__init__(); self.ds_val = ds_val; self.w_gap = float(w_gap); self.best = -1e9
        self.best_path = "reports/adv_inproc_fairbest.keras"
    def on_epoch_end(self, epoch, logs=None):
        y_true,y_pred,g_true=[],[],[]
        for xb,yb in self.ds_val:
            py,_ = self.model.predict(xb, verbose=0)
            y_pred += list(np.argmax(py,1))
            y_true += list(np.argmax(yb["emotion"].numpy(),1))
            g_true  += list(np.argmax(yb["skin"].numpy(),1))
        y_true,y_pred,g_true = map(np.array,(y_true,y_pred,g_true))
        acc = accuracy_score(y_true,y_pred)
        # Light/Dark gap
        gap = abs(
          accuracy_score(y_true[g_true==0], y_pred[g_true==0]) -
          accuracy_score(y_true[g_true==1], y_pred[g_true==1])
        )
        score = acc - self.w_gap*gap
        print(f"[fairsel] acc={acc:.4f} gap={gap:.4f} score={score:.4f}")
        if score > self.best:
            self.best = score; self.model.save(self.best_path); print("[fairsel] saved fair-best")


class GradientReversal(layers.Layer):
    def __init__(self, lamb=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lamb = float(lamb)

    def call(self, x):
        lamb = self.lamb  # catturato nella closure

        @tf.custom_gradient
        def _flip_grad(x):
            def grad(dy):
                # gradiente invertito e scalato
                return -lamb * dy
            return x, grad

        return _flip_grad(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lamb": self.lamb})
        return cfg

class LambdaSchedule(callbacks.Callback):
    def __init__(self, layer_name="grl", max_lambda=1.0, ramp_epochs=8):
        super().__init__()
        self.layer_name = layer_name
        self.max_lambda = float(max_lambda)
        self.ramp_epochs = int(ramp_epochs)
    # def on_epoch_begin(self, epoch, logs=None):
    #     t = min(1.0, (epoch + 1) / max(1, self.ramp_epochs))
    #     lam = self.max_lambda * (2/(1+np.exp(-12*(t-0.5))) - 1)
    #     grl = self.model.get_layer(self.layer_name)
    #     grl.lamb = float(lam)   
    #     print(f"[sched] GRL λ = {grl.lamb:.3f}")
        
    def on_epoch_begin(self, epoch, logs=None):
        # progress t in [0,1] sui ramp_epochs
        t = min(1.0, (epoch + 1) / max(1, self.ramp_epochs))
        # sigmoide monotona 0→max (mai negativa)
        lam = self.max_lambda * (1.0 / (1.0 + np.exp(-12*(t - 0.5))))
        grl = self.model.get_layer(self.layer_name)
        grl.lamb = float(lam)
        print(f"[sched] GRL λ = {grl.lamb:.3f}")


# ---------- model ----------
def build_model(img_size, num_classes, num_groups, lambda_grl=1.0, train_backbone=False):
    inp = layers.Input(shape=(*img_size,3), name="input")
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inp)
    for l in base.layers:
        l.trainable = train_backbone

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    # shared projection
    h = layers.Dense(256, activation="relu")(x)
    h = layers.Dropout(0.3)(h)

    # emotion head
    logits_y = layers.Dense(num_classes, name="logits_y")(h)
    out_y = layers.Activation("softmax", name="emotion")(logits_y)

    # adversary via GRL
    grl = GradientReversal(lambda_grl, name="grl")(h)
    a = layers.Dense(128, activation="relu")(grl)
    a = layers.Dropout(0.3)(a)
    logits_g = layers.Dense(num_groups, name="logits_g")(a)
    out_g = layers.Activation("softmax", name="skin")(logits_g)

    model = models.Model(inp, [out_y, out_g], name="adv_end2end")
    return model

def unfreeze_last_n_backbone_layers(model, n_last, freeze_bn=True):
    """
    Unfreezes only the last n_last layers of the backbone.
    Qui il backbone è "fuso" nel grafo: assumiamo che tutti i layer
    fino a 'gap' appartengano al backbone MobileNetV2.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization

    # trova l'indice della GAP (è quella che hai chiamato 'gap' in build_model)
    gap_idx = None
    for i, l in enumerate(model.layers):
        if l.name == "gap":
            gap_idx = i
            break
    if gap_idx is None:
        raise RuntimeError("GAP layer 'gap' non trovata. Rinominata?")

    backbone_layers = model.layers[:gap_idx]  # tutti prima della GAP
    total = len(backbone_layers)
    if n_last > total:
        n_last = total

    # Freeze tutto il backbone
    for l in backbone_layers:
        l.trainable = False

    # Unfreeze solo gli ultimi n_last (saltando le BN se richiesto)
    unfrozen = 0
    for l in backbone_layers[-n_last:]:
        if freeze_bn and isinstance(l, BatchNormalization):
            l.trainable = False
        else:
            l.trainable = True
            unfrozen += 1

    print(f"[INFO] Unfroze last {n_last} backbone layers "
          f"({unfrozen} effettivamente trainable; BN tenute freeze={freeze_bn}). "
          f"Backbone size ≈ {total} layers.")


def save_cm(y_true, y_pred, labels, name):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(xticks_rotation=45, colorbar=False)
    plt.title(name); plt.tight_layout()
    path = os.path.join("reports", f"{name}.png")
    plt.savefig(path, dpi=160); plt.close()
    print(f"[saved] {path}")

def main():
    args = parse_args()
    tr_csv = os.path.join(args.splits_dir, "train_split_4.csv")
    va_csv = os.path.join(args.splits_dir, "val_split_4.csv")
    te_csv = os.path.join(args.splits_dir, "test_split_4.csv")
    df_tr, df_va, df_te = load_split_csv(tr_csv), load_split_csv(va_csv), load_split_csv(te_csv)
    le_y, le_g = make_label_encoders(df_tr)
    num_classes, num_groups = len(le_y.classes_), len(le_g.classes_)

    ds_tr = make_ds_balanced(df_tr, args.img_root, le_y, le_g, tuple(args.img_size), args.batch, shuffle=True)
    steps_per_epoch = math.ceil(len(df_tr) / args.batch)
    ds_va = make_ds(df_va, args.img_root, le_y, le_g, tuple(args.img_size), args.batch, shuffle=False)  
    val_steps = math.ceil(len(df_va) / args.batch)
    ds_te = make_ds(df_te, args.img_root, le_y, le_g, tuple(args.img_size), args.batch, shuffle=False)


    # -------- PHASE 1: head-only (backbone frozen) --------
    model = build_model(tuple(args.img_size), num_classes, num_groups, args.lambda_grl, train_backbone=False)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt,
                  loss={"emotion":"categorical_crossentropy","skin":"categorical_crossentropy"},
                  loss_weights={"emotion":1.0,"skin":args.alpha},
                  metrics={"emotion":["accuracy"], "skin":["accuracy"]})
    ckpt = callbacks.ModelCheckpoint("reports/adv_inproc_best.keras",
                                     monitor="val_emotion_accuracy", mode="max",
                                     save_best_only=True, verbose=1)
    es = callbacks.EarlyStopping(monitor="val_emotion_accuracy", mode="max",
                                 patience=5, restore_best_weights=True)

    fairsel = FairSelect(ds_va, w_gap=0.4)

    print("\n[PHASE 1] Frozen backbone (head-only)")
    sched = LambdaSchedule(layer_name="grl", max_lambda=args.lambda_grl, ramp_epochs=max(3, args.freeze_epochs))
    model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=args.freeze_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[ckpt, es, sched, fairsel],
        verbose=2
    )

    # -------- PHASE 2: fine-tuning (unfreeze last N backbone layers) --------
    unfreeze_last_n_backbone_layers(model, args.unfreeze_last)
    # Typically use a smaller LR for FT
    opt_ft = tf.keras.optimizers.Adam(learning_rate=args.lr/5.0)
    model.compile(
        optimizer=opt_ft,
        loss={"emotion":"categorical_crossentropy","skin":"categorical_crossentropy"},
        loss_weights={"emotion":1.0, "skin":args.alpha},
        metrics={"emotion":["accuracy"], "skin":["accuracy"]}
    )
    rlr = callbacks.ReduceLROnPlateau(monitor="val_emotion_loss", factor=0.5, patience=3, mode="min", verbose=1)

    print("\n[PHASE 2] Fine-tuning backbone (last N layers)")
    sched2 = LambdaSchedule(layer_name="grl", max_lambda=args.lambda_grl, ramp_epochs=max(5, args.ft_epochs//2))
    model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=args.ft_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[ckpt, es, rlr, sched2, fairsel],
        verbose=2
    )


    # reload best
    # --- reload best ---
    try:
        model = tf.keras.models.load_model("reports/adv_inproc_fairbest.keras",
            custom_objects={"GradientReversal": GradientReversal})
        print("[OK] loaded fairness-aware best checkpoint")
    except Exception:
        try:
            model = tf.keras.models.load_model("reports/adv_inproc_best.keras",
                custom_objects={"GradientReversal": GradientReversal})
            print("[OK] loaded best checkpoint")
        except Exception as e:
            print("[WARN] could not reload best checkpoint:", e)

    # -------- Evaluation on test --------
    y_true, y_pred = [], []
    g_true, g_pred = [], []
    for xb, yb in ds_te:
        py, pg = model.predict(xb, verbose=0)
        y_pred.extend(np.argmax(py, axis=1))
        g_pred.extend(np.argmax(pg, axis=1))
        y_true.extend(np.argmax(yb["emotion"].numpy(), axis=1))
        g_true.extend(np.argmax(yb["skin"].numpy(), axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    g_true, g_pred = np.array(g_true), np.array(g_pred)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    print("\n=== TEST (Adversarial in-process, two-phase) ===")
    print(f"overall: accuracy={acc:.4f} | macro-F1={f1m:.4f} | weighted-F1={f1w:.4f}")
    print(classification_report(y_true, y_pred, target_names=le_y.classes_))

    # adversary sanity (lower ≈ more skin-invariance; chance ≈ 0.5)
    acc_adv = accuracy_score(g_true, g_pred)
    print(f"[adversary] skin accuracy on test: {acc_adv:.4f}")

    # per-group metrics + gaps
    rows = [{"split":"overall","accuracy":acc,"macro_f1":f1m,"weighted_f1":f1w,"adv_acc":acc_adv}]
    skin_names = le_g.inverse_transform(np.arange(num_groups))
    def save_cm_block(name, idx, tag):
        save_cm(y_true[idx], y_pred[idx], le_y.classes_, f"cm_group_{tag}_adv_inproc")
    for name in skin_names:
        mask = (le_g.inverse_transform(g_true) == name)
        idx = np.where(mask)[0]
        acc_g = accuracy_score(y_true[idx], y_pred[idx])
        f1m_g = f1_score(y_true[idx], y_pred[idx], average="macro")
        f1w_g = f1_score(y_true[idx], y_pred[idx], average="weighted")
        print(f"\n--- Group {name} ---")
        print(f"accuracy={acc_g:.4f} | macro-F1={f1m_g:.4f} | weighted-F1={f1w_g:.4f}")
        print(classification_report(y_true[idx], y_pred[idx], target_names=le_y.classes_))
        rows.append({"split":f"group_{name}","accuracy":acc_g,"macro_f1":f1m_g,"weighted_f1":f1w_g})
        save_cm_block(name, idx, name)

    if set(skin_names) >= {"Light","Dark"}:
        def get(split):
            return next(r for r in rows if r["split"]==split)
        gap_acc = get("group_Light")["accuracy"] - get("group_Dark")["accuracy"]
        gap_f1m = get("group_Light")["macro_f1"] - get("group_Dark")["macro_f1"]
        print("\n=== GAPS (Light − Dark) ===")
        print(f"Δ accuracy = {gap_acc:+.4f}")
        print(f"Δ macro-F1 = {gap_f1m:+.4f}")
        rows.append({"split":"gap_L_minus_D","accuracy":gap_acc,"macro_f1":gap_f1m,"weighted_f1":np.nan})

    save_cm(y_true, y_pred, le_y.classes_, "cm_overall_adv_inproc")
    pd.DataFrame(rows).to_csv("reports/adv_inproc_metrics.csv", index=False)
    print("[saved] reports/adv_inproc_metrics.csv")

if __name__ == "__main__":
    main()
