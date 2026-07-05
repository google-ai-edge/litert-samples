"""Cycle 0: NIMA (idealo MobileNet) reference — build, load weights, score a test image.
Run in ~/tfconv (TF 2.21)."""
import os, sys, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "image-quality-assessment")
KIND = sys.argv[1] if len(sys.argv) > 1 else "aesthetic"
W = os.path.join(REPO, "models", "MobileNet", f"weights_mobilenet_{KIND}_0.{'07' if KIND=='aesthetic' else '11'}.hdf5")
IMG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "test.jpg")

def build():
    base = MobileNet(input_shape=(224, 224, 3), weights=None, include_top=False, pooling="avg")
    x = Dropout(0.0)(base.output)
    x = Dense(10, activation="softmax")(x)
    return Model(base.inputs, x)

m = build()
m.load_weights(W)
print("loaded", W)

raw = tf.io.read_file(IMG)
img = tf.image.decode_jpeg(raw, channels=3)
img = tf.image.resize(img, [224, 224], method="bilinear")
x = preprocess_input(img.numpy().astype(np.float32)[None])   # /127.5 - 1
p = m.predict(x, verbose=0)[0]                             # [10]
score = float(np.sum((np.arange(10) + 1) * p))
print(f"[{KIND}] distribution:", np.round(p, 3))
print(f"[{KIND}] mean score (1-10): {score:.3f}")
np.savez(os.path.join(HERE, f"ref_{KIND}.npz"), x=x, p=p, score=score)
print("input range", float(x.min()), float(x.max()))
