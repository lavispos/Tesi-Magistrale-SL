# gradcam_mobilenetv2.py
# Uso:
#   python gradcam_mobilenetv2.py \
#     --csv rafdb_emotions.csv \
#     --ann skin_tone_annotationsHLbinary.csv \
#     --model mobilenet_head.h5 \
#     --outdir gradcam_out \
#     --classes disgust,fear,neutral \
#     --n_per_combo 2 \
#     --layer Conv_1

import os, argparse, cv2, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageDraw, ImageFont

# ======= Classi =======
CLS_ORDER = ['anger','disgust','fear','happiness','neutral','sadness','surprise']
LAB2IDX = {c: i for i, c in enumerate(CLS_ORDER)}
# IDX2LAB: mappa numerica (1..7) -> label string, utile se il CSV usa 1..7
IDX2LAB = {i + 1: c for i, c in enumerate(CLS_ORDER)}

# ======= Utility =======
def load_or_build_model(path_or_weights, num_classes=7, input_shape=(224,224,3)):
    try:
        m = load_model(path_or_weights, compile=False)
        return m
    except Exception:
        base = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=input_shape, weights='imagenet')
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        out = tf.keras.layers.Dense(num_classes, activation='softmax', name='pred')(x)
        m = tf.keras.Model(inputs=base.input, outputs=out)
        m.load_weights(path_or_weights)
        return m

def read_img_tensor(path, size=224):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Immagine non trovata: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = preprocess_input(rgb.astype(np.float32))
    return rgb, np.expand_dims(x, 0)


def _is_conv_layer(layer):
    # supporta Conv2D, DepthwiseConv2D, separable convs ecc.
    return 'Conv' in layer.__class__.__name__ or 'conv' in layer.__class__.__name__



def make_gradcam(model, img_tensor, class_index=None, last_conv_name='Conv_1'):
    """
    Robust Grad-CAM that handles nested backbone Functional models.
    Strategy:
      - if a backbone named 'mobilenetv2_1.00_224' exists, build:
          backbone_sub: backbone.input -> (target_layer.output, backbone.output)
          classifier: Input(shape=backbone.output.shape[1:]) -> ... -> model.output
        then compute conv_map, backbone_out = backbone_sub(img)
        preds = classifier(backbone_out)
        grads = d(preds[class_index]) / d(conv_map)
      - fallback: try to build a direct grad_model(inputs=model.inputs, outputs=[layer.output, model.output])
    """
    # --- step 0: normalize img_tensor -> numpy batch
    if isinstance(img_tensor, tf.Tensor):
        img_np = img_tensor.numpy()
    else:
        img_np = np.asarray(img_tensor, dtype=np.float32)
    if img_np.ndim == 3:
        img_np = np.expand_dims(img_np, 0)
    assert img_np.ndim == 4, f"img_tensor shape inattesa: {img_np.shape}"
    img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)

    # --- try: use backbone submodel if present
    backbone = None
    try:
        backbone = model.get_layer('mobilenetv2_1.00_224')
    except Exception:
        backbone = None

    if backbone is not None:
        # find target layer inside backbone
        target_layer = None
        try:
            target_layer = backbone.get_layer(last_conv_name)
        except Exception:
            # fallback: last conv-like in backbone
            for lyr in reversed(backbone.layers):
                if _is_conv_layer(lyr):
                    target_layer = lyr
                    print(f"[Grad-CAM] Using fallback target inside backbone: '{target_layer.name}'")
                    break
        if target_layer is None:
            raise ValueError("Nessun layer conv trovato nel backbone per Grad-CAM.")

        # build backbone_sub: backbone.input -> (target_layer.output, backbone.output)
        try:
            backbone_sub = tf.keras.Model(inputs=backbone.input, outputs=[target_layer.output, backbone.output])
        except Exception as e:
            raise RuntimeError(f"Impossibile costruire backbone_sub: {e}")

        # build classifier: layers after backbone in the top model
        # find backbone index in model.layers
        idx = None
        for i, lyr in enumerate(model.layers):
            if lyr is backbone:
                idx = i
                break
        if idx is None:
            # fallback: try to build classifier from model by using backbone.output shape
            raise RuntimeError("Backbone trovato ma non riesco a localizzarlo nella lista delle layer del modello principale.")

        # layers after backbone
        layers_after = model.layers[idx+1:]
        # build classifier model that maps backbone.output -> final predictions
        feat_shape = backbone.output.shape[1:]
        feat_inp = tf.keras.Input(shape=tuple([int(s) for s in feat_shape]))
        x = feat_inp
        for lyr in layers_after:
            # some layers might be nested Models — calling them is fine in general
            x = lyr(x)
        classifier = tf.keras.Model(inputs=feat_inp, outputs=x)

        # now forward: conv_map, backbone_out from backbone_sub; preds from classifier(backbone_out)
        with tf.GradientTape() as tape:
            conv_out, backbone_out = backbone_sub(img_tf)   # conv_out: (B,h,w,c)
            tape.watch(conv_out)
            preds = classifier(backbone_out, training=False)
            if class_index is None:
                class_index = int(tf.argmax(preds[0]).numpy())
            class_channel = preds[:, class_index]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            raise RuntimeError("Gradiente nullo (backbone path). Controlla class_index e modello.")

        # compute weights and cam
        weights = tf.reduce_mean(grads, axis=(1, 2))  # (B, c)
        convmap = conv_out[0]                         # (h,w,c)
        wts = weights[0]
        heat = tf.reduce_sum(convmap * wts, axis=-1)
        heat = tf.nn.relu(heat)
        maxv = tf.reduce_max(heat)
        if maxv.numpy() > 0:
            heat = heat / (maxv + 1e-8)
        return heat.numpy(), int(class_index)

    # --- fallback: try direct grad_model (previous approach)
    # try to find a conv layer on top-level model
    target_layer = None
    try:
        target_layer = model.get_layer(last_conv_name)
    except Exception:
        for lyr in reversed(model.layers):
            if _is_conv_layer(lyr):
                target_layer = lyr
                break
    if target_layer is None:
        raise ValueError("Nessun layer conv trovato per Grad-CAM (fallback).")

    # try to build direct grad_model; if fails, raise
    try:
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
    except Exception as e:
        raise RuntimeError(f"Impossibile costruire grad_model diretto: {e}")

    # compute 
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tf, training=False)
        tape.watch(conv_out)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]).numpy())
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_out)
    if grads is None:
        raise RuntimeError("Gradiente nullo (direct fallback).")
    weights = tf.reduce_mean(grads, axis=(1, 2))
    convmap = conv_out[0]
    wts = weights[0]
    heat = tf.reduce_sum(convmap * wts, axis=-1)
    heat = tf.nn.relu(heat)
    maxv = tf.reduce_max(heat)
    if maxv.numpy() > 0:
        heat = heat / (maxv + 1e-8)
    return heat.numpy(), int(class_index)



def overlay_heatmap(rgb, heatmap, alpha=0.35):
    hmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    hmap = np.uint8(255 * np.clip(hmap, 0, 1))
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    out = cv2.addWeighted(hmap, alpha, rgb, 1 - alpha, 0)
    return out

def predict_batch(model, paths, size=224, batch=64):
    xs, idxs = [], []
    preds_all = [None] * len(paths)
    for i, p in enumerate(paths):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        xs.append(preprocess_input(rgb.astype(np.float32)))
        idxs.append(i)
        if len(xs) == batch or i == len(paths) - 1:
            if not xs:
                continue
            X = np.stack(xs, 0)
            pr = model.predict(X, verbose=0)
            for j, ii in enumerate(idxs):
                preds_all[ii] = pr[j]
            xs, idxs = [], []
    return preds_all

def make_grid(tiles, titles, save_path, cols=4, pad=8, bg=(255,255,255), title_h=26):
    if len(tiles) == 0:
        # crea comunque un'immagine vuota minimal per segnalare "nessun risultato"
        Image.new('RGB', (200, 50), bg).save(save_path)
        return
    H, W, _ = tiles[0].shape
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new('RGB', (cols * W + (cols + 1) * pad, rows * (H + title_h) + (rows + 1) * pad), bg)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for idx, (img, txt) in enumerate(zip(tiles, titles)):
        r, c = divmod(idx, cols)
        x0 = pad + c * (W + pad); y0 = pad + r * (H + title_h + pad)
        grid.paste(Image.fromarray(img), (x0, y0))
        draw.text((x0, y0 + H + 4), txt[:40], fill=(0,0,0), font=font)
    grid.save(save_path)

# ======= Main =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV base: image_name, image_path, emotion_label (+ opz. skin_tone_HL/_binary)')
    ap.add_argument('--ann', default=None, help='CSV annotazioni: contiene skin_tone_HL_binary o skin_tone_HL')
    ap.add_argument('--model', required=True, help='Percorso modello (.h5 / .keras) o pesi')
    ap.add_argument('--outdir', default='gradcam_out')
    ap.add_argument('--classes', default='disgust,fear,neutral', help='classi target separate da virgola')
    ap.add_argument('--n_per_combo', type=int, default=2, help='campioni per (classe × esito × gruppo)')
    ap.add_argument('--layer', default='Conv_1', help='nome dell’ultimo layer conv')
    ap.add_argument('--img_size', type=int, default=224)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    model = load_or_build_model(args.model, num_classes=7, input_shape=(args.img_size, args.img_size, 3))

    # --- Carica CSV base
    df = pd.read_csv(args.csv)
    base_required = {'image_name','image_path','emotion_label'}
    miss = base_required - set(df.columns)
    if miss:
        raise ValueError(f"Mancano dal CSV base: {miss}")
    # Se emotion_label è numerica (1..7), mappa a stringhe
    try:
        if pd.api.types.is_integer_dtype(df['emotion_label']) or set(df['emotion_label'].unique()) <= set(range(1,8)):
            df['emotion_label'] = df['emotion_label'].map(IDX2LAB)
    except Exception:
        # no-op: mantieni come sono
        pass

    # --- Assicura la colonna binaria (merge qui se serve)
    if 'skin_tone_HL_binary' not in df.columns:
        if 'skin_tone_HL' in df.columns:
            df['skin_tone_HL_binary'] = df['skin_tone_HL'].map({'Light':'Light','Medium':'Light','Dark':'Dark'})
        else:
            if args.ann is None:
                raise ValueError("Serve --ann per fare il merge con le annotazioni di skin tone.")
            ann = pd.read_csv(args.ann)
            if 'skin_tone_HL_binary' in ann.columns:
                m = ann[['image_name','skin_tone_HL_binary']].drop_duplicates('image_name')
            elif 'skin_tone_HL' in ann.columns:
                m = ann[['image_name','skin_tone_HL']].drop_duplicates('image_name')
                m['skin_tone_HL_binary'] = m['skin_tone_HL'].map({'Light':'Light','Medium':'Light','Dark':'Dark'})
                m = m[['image_name','skin_tone_HL_binary']]
            else:
                raise ValueError("Nel file --ann manca sia 'skin_tone_HL_binary' sia 'skin_tone_HL'.")
            df = df.merge(m, on='image_name', how='left')

    # Normalizza e filtra
    df = df.drop_duplicates(subset='image_name', keep='first').reset_index(drop=True)
    df['skin_tone_HL_binary'] = df['skin_tone_HL_binary'].astype(str).str.strip().str.title()
    df = df[df['skin_tone_HL_binary'].isin(['Light','Dark'])].dropna(subset=['image_path']).reset_index(drop=True)
    # Filtra solo classi supportate
    target_classes = [c.strip() for c in args.classes.split(',') if c.strip() in CLS_ORDER]
    df = df[df['emotion_label'].isin(CLS_ORDER)].copy()

    # Per una prova veloce limita il numero di immagini (quick debug):
    # commenta/discommenta questa riga per test rapidi (es. 40)
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(40)

    # Predizioni
    paths = df['image_path'].tolist()
    preds = predict_batch(model, paths, size=args.img_size, batch=64)
    pred_idx = [None if p is None else int(np.argmax(p)) for p in preds]
    pred_lab = [None if i is None else CLS_ORDER[i] for i in pred_idx]
    df['pred_label'] = pred_lab
    df['pred_ok'] = (df['pred_label'] == df['emotion_label'])

    # Selezione bilanciata: gruppi × esito × classi
        # Selezione bilanciata: gruppi × esito × classi
    tiles_light, titles_light = [], []
    tiles_dark,  titles_dark  = [], []

    rng = np.random.default_rng(42)
    used_by_group = {'Light': set(), 'Dark': set()}

    def draw_unique_from_pool(pool_idx, k, used_set):
        avail = [i for i in pool_idx if i not in used_set]
        take = min(k, len(avail))
        if take == 0:
            return []
        chosen = list(rng.choice(avail, size=take, replace=False))
        used_set.update(chosen)
        return chosen

    for cls in target_classes:
        for ok_flag in [True, False]:
            # LIGHT
            group = 'Light'
            sub_all = df[(df['skin_tone_HL_binary'] == group) & (df['emotion_label'] == cls)]
            if len(sub_all) == 0:
                rows_light_idx = []
            else:
                sub_ok = sub_all[sub_all['pred_ok'] == ok_flag] if 'pred_ok' in df.columns else sub_all
                pool_idx = sub_ok.index.values if len(sub_ok) > 0 else sub_all.index.values
                rows_light_idx = draw_unique_from_pool(pool_idx, args.n_per_combo, used_by_group[group])

            for ii in rows_light_idx:
                r = df.loc[ii]
                try:
                    rgb, x = read_img_tensor(r.image_path, size=args.img_size)
                    # Qui scegli se vuoi spiegare la predizione (pred_label) o il ground-truth (emotion_label)
                    heat, _ = make_gradcam(model, x, class_index=LAB2IDX.get(r['pred_label'], None), last_conv_name=args.layer)
                    tiles_light.append(overlay_heatmap(rgb, heat, alpha=0.38))
                    titles_light.append(f"{cls} | {'OK' if ok_flag else 'ERR'} | pred={r['pred_label']}")
                except Exception as e:
                    print(f"[WARN] Light sample skipped ({r.image_path}): {e}")

            # DARK
            group = 'Dark'
            sub_all = df[(df['skin_tone_HL_binary'] == group) & (df['emotion_label'] == cls)]
            if len(sub_all) == 0:
                rows_dark_idx = []
            else:
                sub_ok = sub_all[sub_all['pred_ok'] == ok_flag] if 'pred_ok' in df.columns else sub_all
                pool_idx = sub_ok.index.values if len(sub_ok) > 0 else sub_all.index.values
                rows_dark_idx = draw_unique_from_pool(pool_idx, args.n_per_combo, used_by_group[group])

            for ii in rows_dark_idx:
                r = df.loc[ii]
                try:
                    rgb, x = read_img_tensor(r.image_path, size=args.img_size)
                    heat, _ = make_gradcam(model, x, class_index=LAB2IDX.get(r['pred_label'], None), last_conv_name=args.layer)
                    tiles_dark.append(overlay_heatmap(rgb, heat, alpha=0.38))
                    titles_dark.append(f"{cls} | {'OK' if ok_flag else 'ERR'} | pred={r['pred_label']}")
                except Exception as e:
                    print(f"[WARN] Dark sample skipped ({r.image_path}): {e}")



    #def take_samples(group_name, ok_flag, cls, n):
        # prova a prendere le righe che corrispondono anche a pred_ok
    #    sub = df[
    #        (df['skin_tone_HL_binary'] == group_name) &
    #        (df['emotion_label'] == cls) &
    #        (df['pred_ok'] == ok_flag)
     #   ]
        # se non ce ne sono, fallback: prendi dalla stessa classe e gruppo ignorando pred_ok
    #    if len(sub) == 0:
    #        sub = df[
    #            (df['skin_tone_HL_binary'] == group_name) &
    #            (df['emotion_label'] == cls)
    #        ]
    #    if len(sub) == 0:
            # nessun esempio disponibile per quel gruppo+classe
    #        return pd.DataFrame(columns=df.columns)
    #    idx = rng.choice(sub.index.values, size=min(n, len(sub)), replace=False)
    #    return df.loc[idx]

#    for cls in target_classes:
#        for ok_flag in [True, False]:
            # Light
#            rows = take_samples('Light', ok_flag, cls, args.n_per_combo)
#            for _, r in rows.iterrows():
#                try:
#                    rgb, x = read_img_tensor(r.image_path, size=args.img_size)
#                    heat, _ = make_gradcam(model, x, class_index=LAB2IDX.get(r['pred_label'], None), last_conv_name=args.layer)
#                    tiles_light.append(overlay_heatmap(rgb, heat, alpha=0.38))
#                    titles_light.append(f"{cls} | {'OK' if ok_flag else 'ERR'} | pred={r['pred_label']}")
#                except Exception as e:
#                    print(f"[WARN] Light sample skipped ({r.image_path}): {e}")
            # Dark
#            rows = take_samples('Dark', ok_flag, cls, args.n_per_combo)
#            for _, r in rows.iterrows():
#                try:
 #                   rgb, x = read_img_tensor(r.image_path, size=args.img_size)
 #                   heat, _ = make_gradcam(model, x, class_index=LAB2IDX.get(r['pred_label'], None), last_conv_name=args.layer)
 #                   tiles_dark.append(overlay_heatmap(rgb, heat, alpha=0.38))
 #                   titles_dark.append(f"{cls} | {'OK' if ok_flag else 'ERR'} | pred={r['pred_label']}")
 #               except Exception as e:
 #                   print(f"[WARN] Dark sample skipped ({r.image_path}): {e}")

    # Salva griglie
    os.makedirs(args.outdir, exist_ok=True)
    save_L = os.path.join(args.outdir, 'gradcam_light_examples.png')
    save_D = os.path.join(args.outdir, 'gradcam_dark_examples.png')
    make_grid(tiles_light, titles_light, save_L, cols=4)
    make_grid(tiles_dark,  titles_dark,  save_D, cols=4)
    print(f"Saved:\n  {save_L}\n  {save_D}")



if __name__ == '__main__':
    main()
