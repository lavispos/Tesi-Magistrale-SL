import tensorflow as tf
m = tf.keras.models.load_model('mobilenet_head.h5', compile=False)
print("Model inputs:", m.inputs)
# prova a ottenere il backbone se esiste
try:
    backbone = m.get_layer('mobilenetv2_1.00_224')
    print("Backbone found:", backbone.name, type(backbone).__name__)
    print("Backbone internal layers (last 60):")
    for l in backbone.layers[-60:]:
        ok = True
        try:
            _ = l.output
        except Exception:
            ok = False
        print(l.name, type(l).__name__, ok)
except Exception as e:
    print("No explicit backbone layer found; printing model layers instead.")
    for l in m.layers[-80:]:
        ok = True
        try:
            _ = l.output
        except Exception:
            ok = False
        print(l.name, type(l).__name__, ok)
