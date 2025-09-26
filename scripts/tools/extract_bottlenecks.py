#!/usr/bin/env python3
# extract_bottlenecks.py
# Usage:
# python3 extract_bottlenecks.py --weights mobilenet_ft_4_full.keras --splits_dir splits --out_dir features --batch 32 --use_preprocess

import os, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator






def make_feature_model_from_trained(weights_path=None, img_size=(224,224)):
    """
    Ricostruisce l'architettura usata in training (MobileNetV2 base + GAP + head),
    prova a caricare un full model .keras se fornito, altrimenti ricostruisce e prova a
    caricare pesi .h5. Ritorna un modello che estrae il bottleneck (output della GAP).
    """
    # If user gave a full model (.keras) try loading it directly
    if weights_path and os.path.exists(weights_path) and weights_path.endswith(".keras"):
        try:
            full_model = tf.keras.models.load_model(weights_path)
            print(f"[OK] Caricato full model da: {weights_path}")
        except Exception as e:
            print(f"[WARN] Impossibile caricare full model .keras: {e}. Farò fallback ricostruendo l'architettura.")
            full_model = None
    else:
        full_model = None

    # If no full_model, build architecture and optionally load weights
    if full_model is None:
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(7, activation="softmax")(x)
        full_model = tf.keras.models.Model(inputs=base.input, outputs=out)

        # warmup to build weights shapes
        _ = full_model.predict(np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32), verbose=0)

        if weights_path and os.path.exists(weights_path):
            try:
                full_model.load_weights(weights_path)
                print(f"[OK] Caricati pesi nel modello ricostruito da: {weights_path}")
            except Exception as e:
                print(f"[WARN] Impossibile caricare i pesi nel modello ricostruito: {e}")
                print("[WARN] Verranno usati i pesi ImageNet di default per il backbone.")

    # Find a GlobalAveragePooling2D layer in full_model
    gap_layer = None
    for layer in full_model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
            break

    # Try to build feature-extractor robustly using the found GAP
    if gap_layer is not None:
        valid_input = None
        # prefer full_model.inputs if available
        try:
            if hasattr(full_model, "inputs") and full_model.inputs:
                valid_input = full_model.inputs[0]
        except Exception:
            valid_input = None

        # fallback to full_model.input
        if valid_input is None and hasattr(full_model, "input"):
            try:
                valid_input = full_model.input
            except Exception:
                valid_input = None

        # fallback to backbone_candidate.input (first layer)
        backbone_candidate = None
        try:
            backbone_candidate = full_model.layers[0]
            if valid_input is None:
                try:
                    valid_input = backbone_candidate.input
                except Exception:
                    valid_input = None
        except Exception:
            backbone_candidate = None

        # Try to build using the gap_layer.output and a valid input tensor
        if valid_input is not None:
            try:
                feat_model = tf.keras.models.Model(inputs=valid_input, outputs=gap_layer.output)
                print("[INFO] Feature extractor costruito usando la GlobalAveragePooling2D del modello caricato.")
                return feat_model
            except Exception as e:
                print(f"[WARN] Tentativo Model(inputs=..., outputs=gap) fallito: {e}. Procedo con fallback.")

    # Fallback: apply a new GAP to the backbone candidate's output and build a model
    try:
        if backbone_candidate is None:
            backbone_candidate = full_model.layers[0]
        feat_output = tf.keras.layers.GlobalAveragePooling2D()(backbone_candidate.output)
        # choose a sensible input tensor
        if hasattr(full_model, "inputs") and full_model.inputs:
            in_tensor = full_model.inputs[0]
        else:
            in_tensor = backbone_candidate.input
        feat_model = tf.keras.models.Model(inputs=in_tensor, outputs=feat_output)
        print("[INFO] Feature extractor costruito applicando GAP al backbone (fallback).")
        return feat_model
    except Exception as e:
        raise RuntimeError("Impossibile costruire il feature extractor (GAP non trovato e fallback fallito): " + str(e))


def make_gen_from_df(df, x_col, target_size=(224,224), batch=32, preprocess=True):
    if preprocess:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col=x_col,
        y_col=None,
        class_mode=None,
        target_size=target_size,
        batch_size=batch,
        shuffle=False
    )
    return gen


def extract_split(split_csv, weights_path, out_path, batch, img_col="image_path", use_preprocess=True, img_size=(224,224), subset=None):
    df = pd.read_csv(split_csv)
    if subset:
        df = df.sample(min(len(df), subset), random_state=42).reset_index(drop=True)
    if img_col not in df.columns:
        raise ValueError(f"Column {img_col} not found in {split_csv}")

    # keep metadata columns we want in output (if present)
    meta_cols = {}
    for c in ["image_path","image_name","skin_bin_str","skin_tone_HL_binary","emotion_label","y_true"]:
        if c in df.columns:
            meta_cols[c] = df[c].values

    print(f"Extracting features for {len(df)} images from {os.path.basename(split_csv)}")
    feat_model = make_feature_model_from_trained(weights_path=weights_path, img_size=img_size)
    gen = make_gen_from_df(df, x_col=img_col, target_size=img_size, batch=batch, preprocess=use_preprocess)
    steps = int(np.ceil(len(df)/batch))
    feats = feat_model.predict(gen, steps=steps, verbose=1)

    # sanity check length
    if feats.shape[0] != len(df):
        print(f"Warning: extracted {feats.shape[0]} features but expected {len(df)}; trimming/padding")
        feats = feats[:len(df)]

    # Save (corregge chiavi duplicate)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {"features": feats, "image_path": df[img_col].values}

    # add meta columns but avoid key collisions
    for k, v in meta_cols.items():
        key = k
        if key in payload:
            key = f"meta_{k}"
            print(f"[WARN] collision on key '{k}' -> saving as '{key}'")
        payload[key] = v

    np.savez_compressed(out_path, **payload)
    print(f"Saved {out_path} (features shape {feats.shape})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to weights/model file (.keras or .h5)")
    p.add_argument("--splits_dir", default="splits", help="directory with train_split_*.csv")
    p.add_argument("--out_dir", default="features", help="where to save .npz files")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--img_size", type=int, nargs=2, default=(224,224))
    p.add_argument("--use_preprocess", action="store_true", help="use mobilenet preprocess_input (default False if not given)")
    p.add_argument("--subset", type=int, default=None, help="optional: extract only a subset (debug)")
    args = p.parse_args()

    weights = args.weights
    splits_dir = args.splits_dir
    out_dir = args.out_dir
    batch = args.batch
    img_size = tuple(args.img_size)
    use_preprocess = bool(args.use_preprocess)

    splits = {
        "train": os.path.join(splits_dir, "train_split_4.csv"),
        "val":   os.path.join(splits_dir, "val_split_4.csv"),
        "test":  os.path.join(splits_dir, "test_split_4.csv"),
    }

    for name, path in splits.items():
        if not os.path.exists(path):
            print(f"Skip {name}: {path} not found")
            continue
        out_path = os.path.join(out_dir, f"bottleneck_{name}_ft4.npz")
        if os.path.exists(out_path):
            print(f"Found existing {out_path} — skipping (delete file to re-extract)")
            continue
        extract_split(path, weights, out_path, batch=batch, img_col="image_path",
                      use_preprocess=use_preprocess, img_size=img_size, subset=args.subset)


if __name__ == "__main__":
    main()
