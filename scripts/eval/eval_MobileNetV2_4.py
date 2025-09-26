# eval_MobileNetV2_4.py
import os, pandas as pd, numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# configurazione (adattare se necessario)
test_csv = "splits/test_split_4.csv"
batch = 32
img_size = (224,224)
candidates = ["mobilenet_ft_4.weights.h5","mobilenet_ft.weights.h5","mobilenet_ft.h5",
              "mobilenet_head_4.weights.h5","mobilenet_head.weights.h5","mobilenet_head.h5"]

if not os.path.exists(test_csv):
    raise SystemExit("File test_split_4.csv non trovato. Genera prima gli split con MobileNetV2_4.py o usa il file splits/.")

df_test = pd.read_csv(test_csv)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = datagen.flow_from_dataframe(df_test, x_col="image_path", y_col="emotion_label",
                                      target_size=img_size, class_mode="categorical",
                                      batch_size=batch, shuffle=False)
# costruisco modello (stessa architettura)
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(img_size[0],img_size[1],3))
model = models.Sequential([base_model, layers.GlobalAveragePooling2D(), layers.Dense(128,activation="relu"),
                           layers.Dropout(0.3), layers.Dense(len(test_gen.class_indices), activation="softmax")])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# trova primo peso disponibile
weights_path = None
for c in candidates:
    if os.path.exists(c):
        weights_path = c
        break
if weights_path is None:
    raise SystemExit("Nessun file di pesi trovato nella cartella. Metti il peso che vuoi testare.")

print("Using weights:", weights_path)
model.load_weights(weights_path)

probs = model.predict(test_gen, verbose=1)
y_pred = probs.argmax(axis=1)
y_true = test_gen.classes
idx_to_label = {v:k for k,v in test_gen.class_indices.items()}
y_true_labels = [idx_to_label[i] for i in y_true]
y_pred_labels = [idx_to_label[i] for i in y_pred]

print("\nClassification report (on test_split_4.csv):")
print(classification_report(y_true_labels, y_pred_labels, digits=4, zero_division=0))

# salva predizioni
out = pd.DataFrame({"image_path": df_test["image_path"], "y_true": y_true_labels, "y_pred": y_pred_labels})
out.to_csv("reports/test_predictions_eval_4.csv", index=False)
print("Saved predictions to reports/test_predictions_eval_4.csv")
