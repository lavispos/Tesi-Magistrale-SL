#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GroupDRO training for FER (MobileNetV2 baseline) - in-process mitigation.
- Warm-up head (backbone frozen)
- Fine-tuning con ultimi N layer sbloccati
- Group DRO con pesi esponenziali per gruppi (Light/Dark)
- Model selection sul worst-group macro-F1 (validation)
"""

import os, json, argparse, random
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pandas.api.types import is_integer_dtype
from tensorflow.keras.layers import BatchNormalization

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def detect_col(df: pd.DataFrame, candidates, fallback=None):
    for c in candidates:
        if c in df.columns:
            return c
    if fallback is not None:
        return fallback
    raise ValueError(f"None of the candidate columns found: {candidates}. Available: {list(df.columns)}")

def load_split_csvs(splits_dir: str, train_csv=None, val_csv=None, test_csv=None):
    """
    Returns (df_train, df_val, df_test).
    Priority:
      1) explicit path (CLI)
      2) <splits_dir>/<kind>.csv
      3) recursive search: '*{kind}*_4.csv' then common patterns
    """
    def pick(kind, explicit):
        if explicit and os.path.exists(explicit):
            print(f"[Splits] Using {kind} = {explicit}")
            return explicit
        cand = os.path.join(splits_dir, f"{kind}.csv")
        if os.path.exists(cand):
            print(f"[Splits] Using {kind} = {cand}")
            return cand
        preferred = sorted(glob(os.path.join(splits_dir, f"**/*{kind}*_4.csv"), recursive=True))
        if preferred:
            print(f"[Splits] Auto-detected {kind} = {preferred[0]}")
            return preferred[0]
        patterns = {
            "train": ["**/*train*.csv"],
            "val":   ["**/*val*.csv", "**/*valid*.csv", "**/*validation*.csv"],
            "test":  ["**/*test*.csv", "**/*eval*.csv"],
        }[kind]
        for P in patterns:
            hits = sorted(glob(os.path.join(splits_dir, P), recursive=True))
            if hits:
                print(f"[Splits] Auto-detected {kind} = {hits[0]}")
                return hits[0]
        raise FileNotFoundError(f"Could not find a CSV for '{kind}' under {splits_dir}.")

    t = pick("train", train_csv)
    v = pick("val",   val_csv)
    te = pick("test", test_csv)
    return pd.read_csv(t), pd.read_csv(v), pd.read_csv(te)

def make_path_series(df: pd.DataFrame, img_root: str, path_col: str):
    paths = df[path_col].astype(str).tolist()
    if img_root:
        paths = [p if os.path.isabs(p) or p.startswith(img_root) else os.path.join(img_root, p) for p in paths]
    return pd.Series(paths)

def decode_and_resize(path, img_size, training):
    img = tf.io.read_file(path)
    # robust decode (jpeg/png)
    try:
        x = tf.image.decode_jpeg(img, channels=3)
    except:
        x = tf.image.decode_png(img, channels=3)
    x = tf.image.resize(x, img_size, method="bilinear")
    x = tf.image.convert_image_dtype(x, tf.float32)  # [0,1]
    if training:
        x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
        x = tf.image.random_saturation(x, lower=0.85, upper=1.15)
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.clip_by_value(x, 0.0, 1.0)
    x = x * 255.0
    x = preprocess_input(x)  # -> [-1, 1] (MobileNetV2)
    return x

def build_dataset(df: pd.DataFrame, img_root: str, img_size, batch, shuffle, path_col, label_col, group_col, repeat=False):
    paths = make_path_series(df, img_root, path_col).values
    labels = df[label_col].astype(np.int32).values
    groups = df[group_col].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, groups))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 10000), reshuffle_each_iteration=True)

    def _map_fn(path, y, g):
        x = decode_and_resize(path, img_size, training=shuffle)
        return x, y, g

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds

def build_balanced_dataset(df, img_root, img_size, batch, path_col, label_col, group_col):
    """Bilanciamento solo per gruppo (Light/Dark)."""
    dfs = [df[df[group_col]==g] for g in sorted(df[group_col].unique())]
    dsets = []
    for dfg in dfs:
        paths = make_path_series(dfg, img_root, path_col).values
        labels = dfg[label_col].astype(np.int32).values
        groups = dfg[group_col].astype(np.int32).values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, groups))
        ds = ds.shuffle(buffer_size=min(len(dfg), 10000), reshuffle_each_iteration=True)
        ds = ds.map(lambda p,y,g: (decode_and_resize(p, img_size, training=True), y, g),
                    num_parallel_calls=tf.data.AUTOTUNE)
        dsets.append(ds.repeat())
    ds = tf.data.Dataset.sample_from_datasets(dsets, weights=[1/len(dsets)]*len(dsets))
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def build_group_class_balanced_dataset(df, img_root, img_size, batch,
                                       path_col, label_col, group_col):
    """Bilanciamento uniforme sulle celle (gruppo, classe)."""
    uniq_g = sorted(df[group_col].unique().tolist())
    uniq_y = sorted(df[label_col].unique().tolist())
    dsets = []
    for g in uniq_g:
        for y in uniq_y:
            dfg = df[(df[group_col]==g) & (df[label_col]==y)]
            if len(dfg)==0:
                continue
            paths  = make_path_series(dfg, img_root, path_col).values
            labels = dfg[label_col].astype(np.int32).values
            groups = dfg[group_col].astype(np.int32).values
            ds = tf.data.Dataset.from_tensor_slices((paths, labels, groups))
            ds = ds.shuffle(buffer_size=min(len(dfg), 10000), reshuffle_each_iteration=True)
            ds = ds.map(lambda p,y,g: (decode_and_resize(p, img_size, training=True), y, g),
                        num_parallel_calls=tf.data.AUTOTUNE)
            dsets.append(ds.repeat())
    if not dsets:
        raise ValueError("No (group,class) cells to balance. Check columns.")
    ds = tf.data.Dataset.sample_from_datasets(dsets, weights=[1/len(dsets)]*len(dsets))
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------------
# Metrics (macro-F1)
# ----------------------------
def macro_f1_from_arrays(y_true, y_pred, n_classes=7):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = (2*prec*rec) / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

def weighted_f1_from_arrays(y_true, y_pred, n_classes=7):
    """F1 ponderato per supporto di classe."""
    N = len(y_true)
    if N == 0:
        return float("nan")
    f1_sum = 0.0
    supp_sum = 0
    for c in range(n_classes):
        yt_c = (y_true == c)
        yp_c = (y_pred == c)
        tp = np.sum(yt_c & yp_c)
        fp = np.sum(~yt_c & yp_c)
        fn = np.sum(yt_c & ~yp_c)
        supp = np.sum(yt_c)
        if supp == 0:
            continue
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = (2 * prec * rec) / (prec + rec + 1e-12)
        f1_sum += f1 * supp
        supp_sum += supp
    return float(f1_sum / (supp_sum + 1e-12))


def group_metrics(y_true, y_pred, groups, n_classes=7):
    out = {}
    for gval, gname in [(0, "Light"), (1, "Dark")]:
        m = (groups == gval)
        if np.sum(m) == 0:
            out[gname] = {"acc": np.nan, "macro_f1": np.nan, "n": 0}
            continue
        yt = y_true[m]; yp = y_pred[m]
        acc = float(np.mean(yt == yp))
        mf1 = macro_f1_from_arrays(yt, yp, n_classes)
        out[gname] = {"acc": acc, "macro_f1": mf1, "n": int(np.sum(m))}
    return out

# ----------------------------
# Optim & BN helpers
# ----------------------------
def make_optimizer(lr: float, wd: float):
    """Use AdamW if available; otherwise Adam (ignoring wd)."""
    AdamW = getattr(tf.keras.optimizers, "AdamW", None)
    if AdamW is not None:
        if wd and wd > 0.0:
            print(f"[Optim] Using AdamW (lr={lr}, weight_decay={wd})")
            return AdamW(learning_rate=lr, weight_decay=wd)
        print(f"[Optim] Using AdamW (lr={lr}, weight_decay=0)")
        return AdamW(learning_rate=lr, weight_decay=0.0)
    if wd and wd > 0.0:
        print("[WARN] AdamW non disponibile: uso Adam e ignoro weight_decay.")
    print(f"[Optim] Using Adam (lr={lr})")
    return tf.keras.optimizers.Adam(learning_rate=lr)

def freeze_batchnorm(model):
    n = 0
    for l in model.layers:
        if isinstance(l, BatchNormalization):
            l.trainable = False
            n += 1
    print(f"[FreezeBN] Frozen {n} BatchNorm layers.")

# ----------------------------
# Model: GroupDRO wrapper
# ----------------------------
@tf.keras.utils.register_keras_serializable(package="gdro")
class GroupDROModel(tf.keras.Model):
    def __init__(self, backbone, eta=0.2, label_smoothing=0.1):
        super().__init__()
        self.backbone = backbone
        self.eta = float(eta)
        self.label_smoothing = float(label_smoothing)
        # CategoricalCE su one-hot (smoothing gestito qui)
        self.ce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )
        self.ema_L = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.ema_D = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, x, training=False):
        return self.backbone(x, training=training)

    @staticmethod
    def _safe_mean(values, mask):
        num = tf.reduce_sum(values * mask)
        den = tf.reduce_sum(mask) + 1e-8
        return num / den

    def _smooth_one_hot(self, y_int, num_classes):
        y_oh = tf.one_hot(y_int, num_classes, dtype=tf.float32)
        if self.label_smoothing > 0.0:
            eps = tf.constant(self.label_smoothing, tf.float32)
            numc = tf.cast(num_classes, tf.float32)
            y_oh = (1.0 - eps) * y_oh + eps / numc
        return y_oh

    def train_step(self, data):
        x, y_int, g = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)                      # (B,C) softmax
            num_classes = tf.shape(y_pred)[-1]
            y_true = self._smooth_one_hot(y_int, num_classes)    # (B,C)
            loss_vec = self.ce(y_true, y_pred)                   # (B,)

            mask_L = tf.cast(tf.equal(g, 0), tf.float32)
            mask_D = tf.cast(tf.equal(g, 1), tf.float32)
            L_L = self._safe_mean(loss_vec, mask_L)
            L_D = self._safe_mean(loss_vec, mask_D)

            self.ema_L.assign(0.9 * self.ema_L + 0.1 * L_L)
            self.ema_D.assign(0.9 * self.ema_D + 0.1 * L_D)
            L_L_eff = tf.where(tf.reduce_sum(mask_L) > 0, L_L, self.ema_L)
            L_D_eff = tf.where(tf.reduce_sum(mask_D) > 0, L_D, self.ema_D)

            w = tf.nn.softmax(self.eta * tf.stack([L_L_eff, L_D_eff]))  # [w_L, w_D]
            loss = w[0] * L_L + w[1] * L_D
            if self.losses:
                loss += tf.add_n(self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(y_int, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs.update({"loss": loss, "L_L": L_L, "L_D": L_D, "w_L": w[0], "w_D": w[1]})
        return logs

    def test_step(self, data):
        x, y_int, g = data
        y_pred = self(x, training=False)
        num_classes = tf.shape(y_pred)[-1]
        y_true = self._smooth_one_hot(y_int, num_classes)
        loss_vec = self.ce(y_true, y_pred)

        mask_L = tf.cast(tf.equal(g, 0), tf.float32)
        mask_D = tf.cast(tf.equal(g, 1), tf.float32)
        L_L = self._safe_mean(loss_vec, mask_L)
        L_D = self._safe_mean(loss_vec, mask_D)
        loss = tf.reduce_mean(loss_vec)

        self.compiled_metrics.update_state(y_int, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs.update({"loss": loss, "L_L": L_L, "L_D": L_D})
        return logs

# ----------------------------
# Callback: early stop on worst-group macro-F1 (validation)
# ----------------------------
class WorstGroupF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, n_classes, patience=5, out_model="best_gdro.keras",
                 log_weighted_f1=True):
        super().__init__()
        self.val_ds = val_ds
        self.n_classes = n_classes
        self.patience = patience
        self.best = -np.inf
        self.wait = 0
        self.out_model = out_model
        self.log_weighted_f1 = log_weighted_f1

    def on_epoch_end(self, epoch, logs=None):
        y_true_all, y_pred_all, g_all = [], [], []
        for xb, yb, gb in self.val_ds:
            pb = self.model(xb, training=False).numpy()
            y_pred_all.append(np.argmax(pb, axis=1))
            y_true_all.append(yb.numpy())
            g_all.append(gb.numpy())

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        groups = np.concatenate(g_all)

        # Metriche per gruppo
        metrics = group_metrics(y_true, y_pred, groups, self.n_classes)
        light_f1 = metrics["Light"]["macro_f1"]
        dark_f1  = metrics["Dark"]["macro_f1"]

        # >>> DEFINISCI worst PRIMA DI USARLA <<<
        worst = float(np.nanmin([light_f1, dark_f1]))

        # (Opzionale) Weighted-F1 per log
        if self.log_weighted_f1:
            overall_wf1 = weighted_f1_from_arrays(y_true, y_pred, n_classes=self.n_classes)
            mask_L = (groups == 0)
            mask_D = (groups == 1)
            light_wf1 = weighted_f1_from_arrays(y_true[mask_L], y_pred[mask_L], n_classes=self.n_classes)
            dark_wf1  = weighted_f1_from_arrays(y_true[mask_D], y_pred[mask_D], n_classes=self.n_classes)
            print(
                f"\n[VAL] Light macro-F1={light_f1:.4f}, Dark macro-F1={dark_f1:.4f} "
                f"-> Worst={worst:.4f} | W-F1 overall={overall_wf1:.4f}, L={light_wf1:.4f}, D={dark_wf1:.4f}"
            )
        else:
            print(f"\n[VAL] Light macro-F1={light_f1:.4f}, Dark macro-F1={dark_f1:.4f} -> Worst={worst:.4f}")

        # Early stopping / model selection sul worst-group macro-F1
        if worst > self.best:
            self.best = worst
            self.wait = 0
            self.model.backbone.save(self.out_model)
            print(f"[ModelSelection] Saved best model to {self.out_model} (worst-group F1 improved).")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"[EarlyStopping] No improvement for {self.patience} epochs. Stopping.")
                self.model.stop_training = True


# ----------------------------
# Build backbone
# ----------------------------
def build_backbone(input_shape=(224,224,3), n_classes=7, dropout=0.0):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(base.input, out, name="mobilenetv2_fer")
    return model, base

def freeze_all(base):
    for l in base.layers:
        l.trainable = False

def unfreeze_last_n(base, n):
    n = int(n)
    for l in base.layers[:-n]:
        l.trainable = False
    for l in base.layers[-n:]:
        l.trainable = True

# ----------------------------
# Main
# ----------------------------
def main(args):
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    set_seed(args.seed)

    # Load CSVs
    df_tr, df_va, df_te = load_split_csvs(args.splits_dir, args.train_csv, args.val_csv, args.test_csv)

    # skin_bin_str (Light/Dark) -> HL_binary (0/1) se presente
    if "skin_bin_str" in df_tr.columns:
        def map_group(df):
            df = df[df["skin_bin_str"].isin(["Light","Dark"])].copy()
            df["HL_binary"] = (df["skin_bin_str"] == "Dark").astype(np.int32)
            return df
        df_tr, df_va, df_te = map_group(df_tr), map_group(df_va), map_group(df_te)

    # Detect columns
    path_col  = args.path_col  or detect_col(
        df_tr, ["filepath","path","img_path","image_path","file_path","path_rel","relative_path"])
    label_col = args.label_col or detect_col(
        df_tr, ["emotion","label","class","y","target","emotion_idx","emotion_label"])
    group_col = args.group_col or detect_col(
        df_tr, ["HL_binary","skin_bin_str","group","skin","skin_group","HL","skin_tone_bin","tone"])

    # Normalizza etichette in interi
    all_labels = pd.concat([df_tr[label_col], df_va[label_col], df_te[label_col]], ignore_index=True)

    def _try_cast_int(df, col):
        try:
            df[col] = df[col].astype(int)
            return True
        except Exception:
            return False

    if is_integer_dtype(all_labels):
        print("[Labels] Detected integer labels. No mapping needed.")
    else:
        ok_tr = _try_cast_int(df_tr, label_col)
        ok_va = _try_cast_int(df_va, label_col)
        ok_te = _try_cast_int(df_te, label_col)
        if ok_tr and ok_va and ok_te:
            print("[Labels] Casted digit-strings -> int successfully.")
        else:
            classes = sorted(all_labels.dropna().astype(str).unique().tolist())
            class2id = {c:i for i,c in enumerate(classes)}
            for _df in (df_tr, df_va, df_te):
                _df[label_col] = _df[label_col].astype(str).map(class2id).astype(np.int32)
            print(f"[Labels] Mapped string labels -> ids: {class2id}")

    # N classi dai dati
    n_classes_data = int(pd.concat([df_tr[label_col], df_va[label_col], df_te[label_col]]).nunique())
    if args.n_classes and args.n_classes != n_classes_data:
        print(f"[WARN] --n_classes={args.n_classes} differs from data ({n_classes_data}). Using {n_classes_data}.")
    n_classes = n_classes_data

    # Build datasets
    img_size = (args.img_size[0], args.img_size[1])

    if args.balance_group_class:
        ds_tr = build_group_class_balanced_dataset(
            df_tr, args.img_root, img_size, args.batch,
            path_col=path_col, label_col=label_col, group_col=group_col
        )
    elif args.balance_groups:
        ds_tr = build_balanced_dataset(
            df_tr, args.img_root, img_size, args.batch,
            path_col=path_col, label_col=label_col, group_col=group_col
        )
    else:
        ds_tr = build_dataset(
            df_tr, args.img_root, img_size, args.batch, shuffle=True,
            path_col=path_col, label_col=label_col, group_col=group_col
        )

    # dataset bilanciati sono "infiniti" → servono gli steps
    steps_per_epoch = None
    if args.balance_groups or args.balance_group_class:
        steps_per_epoch = int(np.ceil(len(df_tr) / args.batch))

    ds_va = build_dataset(df_va, args.img_root, img_size, args.batch, shuffle=False,
                          path_col=path_col, label_col=label_col, group_col=group_col)
    ds_te = build_dataset(df_te, args.img_root, img_size, args.batch, shuffle=False,
                          path_col=path_col, label_col=label_col, group_col=group_col)

    # Build model
    backbone, base = build_backbone(input_shape=(img_size[0], img_size[1], 3),
                                    n_classes=n_classes, dropout=args.dropout)
    model = GroupDROModel(backbone, eta=args.eta, label_smoothing=args.label_smoothing)

    # ---------------- Warm-up (testa soltanto) ----------------
    freeze_all(base)
    if args.freeze_bn:
        freeze_batchnorm(base)
    model.compile(optimizer=make_optimizer(args.warmup_lr, args.weight_decay),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    if args.freeze_epochs > 0:
        print(f"[Warm-up] epochs={args.freeze_epochs}, lr={args.warmup_lr}")
        model.fit(ds_tr, validation_data=ds_va,
                  epochs=args.freeze_epochs,
                  steps_per_epoch=steps_per_epoch,
                  verbose=1)

    # ---------------- Fine-tuning (sblocca ultimi N) ----------------
    if args.unfreeze_last > 0:
        unfreeze_last_n(base, args.unfreeze_last)
        if args.freeze_bn:
            freeze_batchnorm(base)  # mantieni BN congelate
        print(f"[Fine-tuning] Unfreezing last {args.unfreeze_last} layers.")

    model.compile(optimizer=make_optimizer(args.ft_lr, args.weight_decay),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    cb = WorstGroupF1EarlyStopping(ds_va, n_classes=n_classes,
                                   patience=args.patience, out_model=args.out_model)

    if args.ft_epochs > 0:
        print(f"[Fine-tuning] epochs={args.ft_epochs}, lr={args.ft_lr}")
        model.fit(ds_tr, validation_data=ds_va,
                  epochs=args.ft_epochs,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=[cb], verbose=1)

    # Load best model (by worst-group F1)
    print("[Load best] Loading best model selected on worst-group F1.")
    best = tf.keras.models.load_model(
        args.out_model,
        compile=False,
        custom_objects={"GroupDROModel": GroupDROModel}
    )

    # Ripristina i pesi nel wrapper corrente
    if hasattr(best, "backbone"):
        model.backbone.set_weights(best.backbone.get_weights())
    else:
        model.backbone.set_weights(best.get_weights())

    # ---- Evaluation on TEST ----
    y_true_all, y_pred_all, g_all = [], [], []
    for xb, yb, gb in ds_te:
        pb = model(xb, training=False).numpy()
        y_pred_all.append(np.argmax(pb, axis=1))
        y_true_all.append(yb.numpy())
        g_all.append(gb.numpy())
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    groups = np.concatenate(g_all)

    overall_acc = float(np.mean(y_true == y_pred))
    overall_mf1 = macro_f1_from_arrays(y_true, y_pred, n_classes=n_classes)
    gm = group_metrics(y_true, y_pred, groups, n_classes=n_classes)
    gap_acc  = gm["Light"]["acc"]    - gm["Dark"]["acc"]
    gap_mf1  = gm["Light"]["macro_f1"] - gm["Dark"]["macro_f1"]
    worst_f1 = float(np.nanmin([gm["Light"]["macro_f1"], gm["Dark"]["macro_f1"]]))
    # Weighted-F1 complessivi e per gruppo
    overall_wf1 = weighted_f1_from_arrays(y_true, y_pred, n_classes=n_classes)
    mask_L = (groups == 0)
    mask_D = (groups == 1)
    light_wf1 = weighted_f1_from_arrays(y_true[mask_L], y_pred[mask_L], n_classes=n_classes)
    dark_wf1  = weighted_f1_from_arrays(y_true[mask_D], y_pred[mask_D], n_classes=n_classes)
    # Gap (Light - Dark) anche per Weighted-F1
    gap_wf1 = light_wf1 - dark_wf1
    worst_acc = min(gm["Light"]["acc"], gm["Dark"]["acc"])
        

    print("\n===== TEST RESULTS (GroupDRO) =====")
    print(f"Overall  ACC={overall_acc:.4f}  Macro-F1={overall_mf1:.4f}  Weighted-F1={overall_wf1:.4f}")
    print(f"Light    ACC={gm['Light']['acc']:.4f}  Macro-F1={gm['Light']['macro_f1']:.4f}  Weighted-F1={light_wf1:.4f}  N={gm['Light']['n']}")
    print(f"Dark     ACC={gm['Dark']['acc']:.4f}   Macro-F1={gm['Dark']['macro_f1']:.4f}   Weighted-F1={dark_wf1:.4f}   N={gm['Dark']['n']}")
    print(f"Gap (L-D) ACC={gap_acc:+.4f}  Macro-F1={gap_mf1:+.4f}  Weighted-F1={gap_wf1:+.4f}")
    print(f"Worst-group Macro-F1={worst_f1:.4f}")
    print(f"Worst-group Acc={worst_acc:.4f}")
    print("===================================\n")

    if args.report:
        report = dict(
            overall=dict(
                acc=overall_acc,
                macro_f1=overall_mf1,
                weighted_f1=overall_wf1
            ),
            light=dict(
                acc=gm["Light"]["acc"],
                macro_f1=gm["Light"]["macro_f1"],
                weighted_f1=light_wf1,
                n=gm["Light"]["n"]
            ),
            dark=dict(
                acc=gm["Dark"]["acc"],
                macro_f1=gm["Dark"]["macro_f1"],
                weighted_f1=dark_wf1,
                n=gm["Dark"]["n"]
            ),
            gaps=dict(
                acc=float(gap_acc),
                macro_f1=float(gap_mf1),
                weighted_f1=float(light_wf1 - dark_wf1)
            ),
            worst_group_macro_f1=worst_f1,
            worst_group_acc=worst_acc,
            args=vars(args)
        )
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Saved] Report -> {args.report}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroupDRO mitigation for FER (MobileNetV2).")
    parser.add_argument("--splits_dir", type=str, required=True, help="Dir with train.csv/val.csv/test.csv")
    parser.add_argument("--img_root", type=str, default=".", help="Root dir for image paths in CSVs")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img_size", type=int, nargs=2, default=[224,224])
    parser.add_argument("--n_classes", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--ft_epochs", type=int, default=15)
    parser.add_argument("--unfreeze_last", type=int, default=60)

    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--eta", type=float, default=0.2, help="GroupDRO exponentiated-gradient step")

    parser.add_argument("--patience", type=int, default=5, help="Early stop on worst-group F1 (val).")
    parser.add_argument("--out_model", type=str, default="mobilenet_gdro.keras")
    parser.add_argument("--report", type=str, default="reports/run_gdro.json")

    parser.add_argument("--seed", type=int, default=42)

    # Override nomi colonne / file split
    parser.add_argument("--path_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--group_col", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv",   type=str, default=None)
    parser.add_argument("--test_csv",  type=str, default=None)

    # Regolarizzazioni / sampling
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    mx = parser.add_mutually_exclusive_group()
    mx.add_argument("--balance_groups", action="store_true",
        help="Equal sampling of groups (Light/Dark) in training batches")
    mx.add_argument("--balance_group_class", action="store_true",
        help="Uniform sampling over (group, class) cells in training batches")
    parser.add_argument("--weight_decay", type=float, default=0.0,
        help="Decoupled AdamW weight decay (es. 5e-4 o 1e-3)")
    parser.add_argument("--freeze_bn", action="store_true",
        help="Mantieni le BatchNorm congelate durante warm-up e fine-tuning")

    args = parser.parse_args()
    main(args)
