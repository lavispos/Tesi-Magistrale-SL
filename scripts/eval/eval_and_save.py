# eval_and_save.py
import os, sys, argparse
import pandas as pd, numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CLS_ORDER = ['anger','disgust','fear','happiness','neutral','sadness','surprise']
EMAP = {1:"surprise",2:"fear",3:"disgust",4:"happiness",5:"sadness",6:"anger",7:"neutral"}

def load_and_preprocess(path, size):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return preprocess_input(img.astype('float32'))

def batch_predict(model, paths, size=224, batch=64):
    preds = []
    for i in range(0, len(paths), batch):
        batch_paths = paths[i:i+batch]
        X = []
        for p in batch_paths:
            arr = load_and_preprocess(p, size)
            if arr is None:
                X.append(np.zeros((size,size,3), dtype=np.float32))
            else:
                X.append(arr)
        X = np.stack(X, 0)
        pr = model.predict(X, verbose=0)
        preds.extend(pr)
    return np.array(preds)

def to_idx(lbl):
    # lbl may be numeric 1..7 or string
    try:
        if isinstance(lbl, (int, np.integer)):
            return CLS_ORDER.index(EMAP[int(lbl)])
        if str(lbl).isdigit():
            return CLS_ORDER.index(EMAP[int(lbl)])
    except Exception:
        pass
    s = str(lbl).strip().lower()
    for i,c in enumerate(CLS_ORDER):
        if c.lower() == s:
            return i
    raise ValueError(f"Unknown label: {lbl}")

def evaluate(model_path, csv_path, ann_path, out_prefix, img_size=224):
    print("Loading model:", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Reading CSVs...")
    df = pd.read_csv(csv_path)
    ann = pd.read_csv(ann_path)
    if 'image_name' not in ann.columns:
        ann = ann.rename(columns={ann.columns[0]:'image_name'})
    if 'skin_tone_HL_binary' not in ann.columns and 'skin_tone_HL' in ann.columns:
        ann['skin_tone_HL_binary'] = ann['skin_tone_HL'].map({'Light':'Light','Medium':'Light','Dark':'Dark'})
    df = df.merge(ann[['image_name','skin_tone_HL_binary']].drop_duplicates('image_name'), on='image_name', how='left')
    df['skin_tone_HL_binary'] = df['skin_tone_HL_binary'].astype(str).str.title()
    # filter to known classes
    df = df[df['emotion_label'].notna()].reset_index(drop=True)
    # build true indices and safe paths
    y_true = []
    paths = []
    groups = []
    for _, r in df.iterrows():
        try:
            idx = to_idx(r['emotion_label'])
        except Exception:
            continue
        y_true.append(idx)
        paths.append(r['image_path'])
        groups.append(r.get('skin_tone_HL_binary', 'unknown') if pd.notna(r.get('skin_tone_HL_binary')) else 'unknown')
    y_true = np.array(y_true)
    groups = np.array(groups)
    print("Images to predict:", len(paths))
    preds = batch_predict(model, paths, size=img_size, batch=64)
    y_pred = np.argmax(preds, axis=1)

    # overall metrics
    overall = {
        'n': len(y_true),
        'acc': accuracy_score(y_true, y_pred),
        'wF1': f1_score(y_true, y_pred, average='weighted'),
        'mF1': f1_score(y_true, y_pred, average='macro')
    }
    rows = []
    rows.append(dict(group='overall', **overall))

    # per-group
    for g in ['Light','Dark','unknown']:
        mask = (groups == g)
        if mask.sum() == 0:
            rows.append({'group': g, 'n': 0, 'acc': None, 'wF1': None, 'mF1': None})
            continue
        rows.append({
            'group': g,
            'n': int(mask.sum()),
            'acc': float(accuracy_score(y_true[mask], y_pred[mask])),
            'wF1': float(f1_score(y_true[mask], y_pred[mask], average='weighted')),
            'mF1': float(f1_score(y_true[mask], y_pred[mask], average='macro')),
        })

    df_metrics = pd.DataFrame(rows)
    metrics_csv = f"{out_prefix}_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print("Saved metrics:", metrics_csv)

    # per-class F1 overall and per-group
    perclass = []
    labels = list(range(len(CLS_ORDER)))
    f1_over = f1_score(y_true, y_pred, labels=labels, average=None)
    for i,label in enumerate(CLS_ORDER):
        entry = {'class': label, 'f1_overall': float(f1_over[i])}
        for g in ['Light','Dark']:
            mask = (groups==g)
            if mask.sum()==0:
                entry[f"f1_{g}"] = None
            else:
                f1_g = f1_score(y_true[mask], y_pred[mask], labels=labels, average=None)
                entry[f"f1_{g}"] = float(f1_g[i])
        perclass.append(entry)
    df_pc = pd.DataFrame(perclass)
    perclass_csv = f"{out_prefix}_perclass.csv"
    df_pc.to_csv(perclass_csv, index=False)
    print("Saved per-class F1:", perclass_csv)

    # quick print summary
    print("\nSUMMARY")
    print(df_metrics.to_string(index=False))
    print("\nPer-class F1 (overall):")
    print(df_pc[['class','f1_overall']].to_string(index=False))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--csv', required=True, help='rafdb_emotions.csv (or test split)')
    p.add_argument('--ann', required=True, help='skin_tone_annotationsHLbinary.csv')
    p.add_argument('--out', required=True, help='output prefix for CSVs')
    p.add_argument('--img_size', type=int, default=224)
    args = p.parse_args()
    evaluate(args.model, args.csv, args.ann, args.out, img_size=args.img_size)
