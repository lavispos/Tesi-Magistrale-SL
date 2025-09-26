# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model

os.makedirs("reports", exist_ok=True)

# 1) Scegli il file migliore che hai salvato
WEIGHTS = "mobilenet_ft.h5" if os.path.exists("mobilenet_ft.h5") else "mobilenet_head.h5"
print(f"Loading model from: {WEIGHTS}")

# 2) Carica l'intero modello (niente retraining)
#    (Se avessi salvato SOLO i pesi, qui servirebbe ricostruire l'architettura;
#     ma con ModelCheckpoint .h5 di default hai salvato il modello completo.)
model = tf.keras.models.load_model(WEIGHTS, compile=False)
# === [NEW] Summary e diagramma del SOLO backbone (MobileNetV2) ===
# 1) prova a prendere il primo layer (di solito è il backbone dentro il Sequential)
try:
    base_model = model.layers[0]
    # se per qualche motivo non è un Model annidato, prova per nome
    if not isinstance(base_model, tf.keras.Model):
        base_model = model.get_layer("mobilenetv2_1.00_224")
except Exception:
    # fallback: primo sotto-modello annidato
    base_model = next(l for l in model.layers if isinstance(l, tf.keras.Model))

# 2) salva la summary del backbone su file
with open("reports/backbone_summary.txt", "w") as f:
    base_model.summary(print_fn=lambda s: f.write(s + "\n"))
print("Saved: reports/backbone_summary.txt")

# 3) (opzionale) diagramma del backbone
try:
    plot_model(
        base_model,
        to_file="reports/mobilenetv2_backbone.png",
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=220
    )
    print("Saved: reports/mobilenetv2_backbone.png")
except Exception as e:
    print("plot_model backbone non riuscito:", e)
