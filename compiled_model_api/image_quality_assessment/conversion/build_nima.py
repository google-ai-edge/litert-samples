"""Convert NIMA (idealo MobileNet) aesthetic + technical to fp16 tflite. Run in ~/tfconv."""
import os, sys, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "image-quality-assessment", "models", "MobileNet")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

def build(weights):
    base = MobileNet(input_shape=(224, 224, 3), weights=None, include_top=False, pooling="avg")
    x = Dense(10, activation="softmax")(Dropout(0.0)(base.output))
    m = Model(base.inputs, x); m.load_weights(weights); return m

for kind, tag in [("aesthetic", "07"), ("technical", "11")]:
    m = build(os.path.join(REPO, f"weights_mobilenet_{kind}_0.{tag}.hdf5"))
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    path = os.path.join(OUT, f"nima_{kind}_fp16.tflite")
    open(path, "wb").write(tfl)
    # parity: tflite vs keras on the ref input
    ref = np.load(os.path.join(HERE, f"ref_{kind}.npz"))
    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(ind["index"], ref["x"].astype(np.float32)); it.invoke()
    p = it.get_tensor(outd["index"])[0]
    score = float(np.sum((np.arange(10) + 1) * p))
    corr = float(np.corrcoef(p, ref["p"])[0, 1])
    print(f"[{kind}] {os.path.getsize(path)/1e6:.1f}MB  tflite score {score:.3f} vs ref {float(ref['score']):.3f}  corr {corr:.6f}")
    ref["x"].astype("<f4").tofile(os.path.join(OUT, f"nima_input_{kind}.bin"))
