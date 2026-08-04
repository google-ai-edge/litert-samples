# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TF re-authoring of KittenTTS nano 0.8 (mini-Kokoro / StyleTTS2 + ISTFTNet).

Channels-last throughout ([1, T, C]). Weights come from out/kitten_weights.npz
(dumped from the ONNX initializers). The five BiLSTMs use tf_keras layers so
the TFLite converter emits fused dynamic-length UNIDIRECTIONAL_SEQUENCE_LSTM.

Graph split (host does the duration -> frame expansion between G1 and G2/G3):
    G1 predictor : ids, style, speed -> d[1,N,256], t_en[1,N,128], durations[N]
    G2 prosody   : en[1,T,256], style -> f0[1,2T], n[1,2T]        (BiLSTM inside)
    G4 harmonics : f0[1,2T] -> har[1,600T?,22]  (SineGen + STFT, one pass)
    G3 vocoder   : asr[1,T,128], f0, n, har, style -> wav         (streamable)
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

SD = dict(np.load(OUT / "kitten_weights.npz"))

BERT_LAYERS = 12
BERT_HEADS = 12
BERT_HIDDEN = 768
STYLE_DIM = 128
LRELU = 0.2
SAMPLE_RATE = 24000

LSTM_WEIGHTS = {
    "text_encoder.lstm": ("onnx::LSTM_5652", "onnx::LSTM_5653", "onnx::LSTM_5651"),
    "dur.lstms.0": ("onnx::LSTM_5872", "onnx::LSTM_5873", "onnx::LSTM_5871"),
    "dur.lstms.2": ("onnx::LSTM_5922", "onnx::LSTM_5923", "onnx::LSTM_5921"),
    "predictor.lstm": ("onnx::LSTM_5971", "onnx::LSTM_5972", "onnx::LSTM_5970"),
    "predictor.shared": ("onnx::LSTM_6020", "onnx::LSTM_6021", "onnx::LSTM_6019"),
}

_LSTM_CACHE = {}


def bilstm(key):
    """tf_keras Bidirectional LSTM with ONNX weights (iofc -> ifco reorder)."""
    if key in _LSTM_CACHE:
        return _LSTM_CACHE[key]
    wn, rn, bn = LSTM_WEIGHTS[key]
    W, R, B = SD[wn], SD[rn], SD[bn]  # [2,4H,In], [2,4H,H], [2,8H]
    hidden = R.shape[2]

    def reorder(m):
        # ONNX gates i,o,f,c along axis0 -> keras i,f,c,o
        i, o, f, c = np.split(m, 4, axis=0)
        return np.concatenate([i, f, c, o], axis=0)

    def direction(d):
        kernel = reorder(W[d]).T  # [In, 4H]
        recurrent = reorder(R[d]).T  # [H, 4H]
        bias = reorder(B[d][:4 * hidden]) + reorder(B[d][4 * hidden:])
        return kernel, recurrent, bias

    layer = keras.layers.Bidirectional(
        keras.layers.LSTM(hidden, return_sequences=True), name=key.replace(".", "_"))
    layer.build((1, None, W.shape[2]))
    fk, fr, fb = direction(0)
    bk, br, bb = direction(1)
    layer.set_weights([fk, fr, fb, bk, br, bb])
    _LSTM_CACHE[key] = layer
    return layer


def dense(x, wname, bname=None, transpose=True):
    w = SD[wname]
    w = w.T if transpose else w  # torch Linear weight [out,in] -> [in,out]
    y = tf.matmul(x, tf.constant(w))
    if bname:
        y += tf.constant(SD[bname])
    return y


def conv1d(x, prefix, stride=1, dilation=1, pad="SAME"):
    w = tf.constant(SD[prefix + ".weight"].transpose(2, 1, 0))
    y = tf.nn.conv1d(x, w, stride=stride, padding=pad, dilations=dilation)
    if prefix + ".bias" in SD:
        y += tf.constant(SD[prefix + ".bias"])
    return y


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
    x = (x - mean) * tf.math.rsqrt(var + eps)
    if gamma is not None:
        x = x * gamma + beta
    return x


import json

# ONNX context -> (scale name, bias name): the exporter value-deduplicated the
# InstanceNorm affine params (and even whole IN nodes fed by the same tensor),
# so parameter names cannot be derived from the module path alone
ADAIN_MAP = json.load(open(OUT / "adain_map.json"))
ADAIN_MAP["/F0.0/norm1"] = ADAIN_MAP["/N.0/norm1"]  # CSE-merged with N.0/norm1


def onnx_ctx(prefix, part):
    """'kmodel.predictor.F0.0' + 'norm1' -> '/F0.0/norm1' (ONNX node context)."""
    if prefix.startswith("kmodel.predictor."):
        base = "/" + prefix[len("kmodel.predictor."):]
    elif prefix.startswith("kmodel.decoder.generator."):
        base = "/decoder/generator/" + prefix[len("kmodel.decoder.generator."):]
    elif prefix.startswith("kmodel.decoder."):
        base = "/decoder/" + prefix[len("kmodel.decoder."):]
    else:
        base = "/" + prefix
    return base + "/" + part


def instance_norm(x, ctx, eps=1e-5):
    """Affine InstanceNorm over the time axis, channels-last; params via map."""
    scale_name, bias_name = ADAIN_MAP[ctx][0], ADAIN_MAP[ctx][1]
    mean, var = tf.nn.moments(x, axes=[1], keepdims=True)
    x = (x - mean) * tf.math.rsqrt(var + eps)
    return x * tf.constant(SD[scale_name]) + tf.constant(SD[bias_name])


# ---------------------------------------------------------------- ALBERT bert
def bert(ids):
    """ALBERT mini: 12 shared layers, hidden 768, gelu_new."""
    emb = tf.gather(tf.constant(SD["kmodel.bert.embeddings.word_embeddings.weight"]), ids)
    n = tf.shape(ids)[1]
    pos = tf.constant(SD["kmodel.bert.embeddings.position_embeddings.weight"])[:n]
    tok = tf.constant(SD["kmodel.bert.embeddings.token_type_embeddings.weight"])[0]
    x = emb + pos[None] + tok[None, None]
    x = layer_norm(x, tf.constant(SD["kmodel.bert.embeddings.LayerNorm.weight"]),
                   tf.constant(SD["kmodel.bert.embeddings.LayerNorm.bias"]), eps=1e-12)
    # the ONNX exporter folded the Linear kernels into pre-transposed
    # anonymous initializers (onnx::MatMul_*, [in,out]); biases kept names
    x = dense(x, "onnx::MatMul_5661",
              "kmodel.bert.encoder.embedding_hidden_mapping_in.bias", transpose=False)

    p = "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0."
    kernels = {"attention.query": "onnx::MatMul_5662", "attention.key": "onnx::MatMul_5665",
               "attention.value": "onnx::MatMul_5668", "attention.dense": "onnx::MatMul_5672",
               "ffn": "onnx::MatMul_5673", "ffn_output": "onnx::MatMul_5674"}
    dk = BERT_HIDDEN // BERT_HEADS

    def gelu_new(t):
        return 0.5 * t * (1.0 + tf.tanh(0.7978845608028654 * (t + 0.044715 * t ** 3)))

    for _ in range(BERT_LAYERS):
        q = dense(x, kernels["attention.query"], p + "attention.query.bias", transpose=False)
        k = dense(x, kernels["attention.key"], p + "attention.key.bias", transpose=False)
        v = dense(x, kernels["attention.value"], p + "attention.value.bias", transpose=False)

        def heads(t):
            return tf.transpose(tf.reshape(t, [1, -1, BERT_HEADS, dk]), [0, 2, 1, 3])

        scores = tf.matmul(heads(q), heads(k), transpose_b=True) / dk ** 0.5
        ctx = tf.matmul(tf.nn.softmax(scores, axis=-1), heads(v))
        ctx = tf.reshape(tf.transpose(ctx, [0, 2, 1, 3]), [1, -1, BERT_HIDDEN])
        attn = dense(ctx, kernels["attention.dense"], p + "attention.dense.bias", transpose=False)
        x = layer_norm(x + attn, tf.constant(SD[p + "attention.LayerNorm.weight"]),
                       tf.constant(SD[p + "attention.LayerNorm.bias"]), eps=1e-12)
        h = gelu_new(dense(x, kernels["ffn"], p + "ffn.bias", transpose=False))
        h = dense(h, kernels["ffn_output"], p + "ffn_output.bias", transpose=False)
        x = layer_norm(x + h, tf.constant(SD[p + "full_layer_layer_norm.weight"]),
                       tf.constant(SD[p + "full_layer_layer_norm.bias"]), eps=1e-12)
    return x


def bert_encoder(x):
    return dense(x, "onnx::MatMul_5818", "kmodel.bert_encoder.bias", transpose=False)


# ------------------------------------------------------------- text encoder
def text_encoder(ids):
    x = tf.gather(tf.constant(SD["kmodel.text_encoder.embedding.weight"]), ids)
    for i in range(2):
        x = conv1d(x, f"kmodel.text_encoder.cnn.{i}.0")
        x = layer_norm(x, tf.constant(SD[f"kmodel.text_encoder.cnn.{i}.1.gamma"]),
                       tf.constant(SD[f"kmodel.text_encoder.cnn.{i}.1.beta"]))
        x = tf.nn.leaky_relu(x, LRELU)
    return bilstm("text_encoder.lstm")(x)  # [1,N,128]


# --------------------------------------------------------- duration encoder
def ada_layer_norm(x, style, prefix):
    h = dense(style, prefix + ".fc.weight", prefix + ".fc.bias")  # [1, 2C]
    gamma, beta = tf.split(h, 2, axis=-1)
    return (1.0 + gamma[:, None]) * layer_norm(x) + beta[:, None]


def duration_encoder(d_en, style_p):
    n = tf.shape(d_en)[1]
    s = tf.tile(style_p[:, None], [1, n, 1])  # [1,N,128]
    x = tf.concat([d_en, s], axis=-1)  # [1,N,256]
    x = bilstm("dur.lstms.0")(x)
    x = ada_layer_norm(x, style_p, "kmodel.predictor.text_encoder.lstms.1")
    x = tf.concat([x, s], axis=-1)
    x = bilstm("dur.lstms.2")(x)
    x = ada_layer_norm(x, style_p, "kmodel.predictor.text_encoder.lstms.3")
    return tf.concat([x, s], axis=-1)  # d [1,N,256]


def duration_head(d, speed):
    x = bilstm("predictor.lstm")(d)  # [1,N,128]
    logits = dense(x, "onnx::MatMul_5973",
                   "kmodel.predictor.duration_proj.linear_layer.bias", transpose=False)
    dur = tf.reduce_sum(tf.sigmoid(logits), axis=-1)  # [1,N]
    dur = tf.round(dur[0] / speed)
    return tf.cast(tf.maximum(dur, 1.0), tf.int32)  # [N]


# ------------------------------------------------- AdainResBlk1d (predictor/decoder)
def adain(x, style, prefix, part):
    h = dense(style, f"{prefix}.{part}.fc.weight", f"{prefix}.{part}.fc.bias")
    gamma, beta = tf.split(h, 2, axis=-1)
    xn = instance_norm(x, onnx_ctx(prefix, part))
    return (1.0 + gamma[:, None]) * xn + beta[:, None]


def upsample_nearest2(x):
    return tf.repeat(x, 2, axis=1)


def depthwise_convt_x2(x, prefix):
    """Depthwise ConvTranspose1d(C,1,k=3,s=2,p=1,outpad=1): zero-stuff + conv."""
    w = SD[prefix + ".weight"]  # [C,1,3]
    c = w.shape[0]
    t = tf.shape(x)[1]
    stuffed = tf.reshape(
        tf.concat([x[:, :, None], tf.zeros_like(x)[:, :, None]], axis=2),
        [1, 2 * t, c])
    # convT == conv over the zero-stuffed input with the kernel flipped;
    # SAME (pad 1|1) realizes pad=k-1-p=1 and output_padding=1 lands on the
    # trailing stuffed zero, so the output length is exactly 2T
    filt = tf.constant(np.ascontiguousarray(w[:, 0, ::-1].T)[None, :, :, None])
    y = tf.nn.depthwise_conv2d(stuffed[:, None], filt,
                               strides=[1, 1, 1, 1], padding="SAME")[:, 0]
    return y + tf.constant(SD[prefix + ".bias"])


def adain_resblk(x, style, prefix, upsample=False):
    y = adain(x, style, prefix, "norm1")
    y = tf.nn.leaky_relu(y, LRELU)
    if upsample:
        y = depthwise_convt_x2(y, prefix + ".pool")
    y = conv1d(y, prefix + ".conv1")
    y = adain(y, style, prefix, "norm2")
    y = tf.nn.leaky_relu(y, LRELU)
    y = conv1d(y, prefix + ".conv2")
    sc = upsample_nearest2(x) if upsample else x
    if prefix + ".conv1x1.weight" in SD:
        sc = conv1d(sc, prefix + ".conv1x1")
    return (y + sc) * (2.0 ** -0.5)


# ------------------------------------------------------------------ prosody
def prosody(en, style_p):
    sh = bilstm("predictor.shared")(en)  # [1,T,128]
    f0 = sh
    for i in range(3):
        f0 = adain_resblk(f0, style_p, f"kmodel.predictor.F0.{i}", upsample=(i == 1))
    f0 = conv1d(f0, "kmodel.predictor.F0_proj")  # [1,2T,1]
    n = sh
    for i in range(3):
        n = adain_resblk(n, style_p, f"kmodel.predictor.N.{i}", upsample=(i == 1))
    n = conv1d(n, "kmodel.predictor.N_proj")
    return f0[..., 0], n[..., 0]  # [1,2T]


# ------------------------------------------------------------------- decoder
N_FFT = 20
HOP = 5
UPSAMPLE_SCALE = 300
SINE_AMP = 0.1
VOICED_THRESHOLD = 10.0
HARMONICS = 9


def strided_conv(x, prefix, stride, pad):
    """torch Conv1d(k, stride, pad) == explicit pad + VALID conv."""
    w = tf.constant(SD[prefix + ".weight"].transpose(2, 1, 0))
    x = tf.pad(x, [[0, 0], [pad, pad], [0, 0]])
    y = tf.nn.conv1d(x, w, stride=stride, padding="VALID")
    if prefix + ".bias" in SD:
        y += tf.constant(SD[prefix + ".bias"])
    return y


def convt(x, prefix, k, stride):
    """torch ConvTranspose1d with p=(k-s)//2 == TF SAME transpose conv."""
    w = tf.constant(SD[prefix + ".weight"].transpose(2, 1, 0))  # [k,out,in]
    ch = w.shape[1]
    t = tf.shape(x)[1]
    y = tf.nn.conv1d_transpose(x, w, output_shape=[1, t * stride, ch],
                               strides=stride, padding="SAME")
    return y + tf.constant(SD[prefix + ".bias"])


def resize_linear(x, new_len):
    """ONNX Resize linear + half_pixel over the time axis, channels-last."""
    return tf.image.resize(x[:, :, None, :], [new_len, 1], method="bilinear")[:, :, 0, :]


def harmonics_graph(f0_pred):
    """f0[1,2T] -> har spec+phase [1, 120T+1, 22] (SineGen + STFT, deterministic)."""
    f0 = tf.repeat(f0_pred[..., None], UPSAMPLE_SCALE, axis=1)  # nearest x300 [1,600T,1]
    fn = f0 * tf.constant(np.arange(1, HARMONICS + 1, dtype=np.float32))  # [1,600T,9]
    rad = fn / float(SAMPLE_RATE)
    rad = rad - tf.floor(rad)  # % 1: unvoiced frames have small NEGATIVE f0
    t_up = tf.shape(rad)[1]
    rad_down = resize_linear(rad, t_up // UPSAMPLE_SCALE)  # [1,2T,9]
    phase = tf.cumsum(rad_down, axis=1) * 2.0 * np.pi
    phase_up = resize_linear(phase * float(UPSAMPLE_SCALE), t_up)
    sines = tf.sin(phase_up)  # rand_ini = 0 (deterministic)
    uv = tf.cast(f0 > VOICED_THRESHOLD, tf.float32)  # [1,600T,1]
    sine_waves = sines * SINE_AMP * uv  # noise branch = 0 (deterministic)
    har = tf.tanh(dense(sine_waves, "onnx::MatMul_6116",
                        "kmodel.decoder.generator.m_source.l_linear.bias", transpose=False))
    # CustomSTFT.transform: center pad (edge) + strided conv with DFT bases
    har_pad = tf.concat([tf.repeat(har[:, :1], N_FFT // 2, axis=1), har,
                         tf.repeat(har[:, -1:], N_FFT // 2, axis=1)], axis=1)
    wr = tf.constant(SD["kmodel.decoder.generator.stft.weight_forward_real"].transpose(2, 1, 0))
    wi = tf.constant(SD["kmodel.decoder.generator.stft.weight_forward_imag"].transpose(2, 1, 0))
    re = tf.nn.conv1d(har_pad, wr, stride=HOP, padding="VALID")
    im = tf.nn.conv1d(har_pad, wi, stride=HOP, padding="VALID")
    mag = tf.sqrt(re ** 2 + im ** 2 + 1e-14)
    # atan2 rather than the ONNX half-angle atan: identical angles up to the
    # branch-cut convention on numerically-silent bins (mag ~ 1e-3), where the
    # reference phase is fp chaos anyway; TFLite has a native ATAN2 kernel
    phase_h = tf.atan2(im, re)
    return tf.concat([mag, phase_h], axis=-1)  # [1, 120T+1, 22]


def snake_resblock(x, style, prefix):
    """AdaINResBlock1: snake activations, kernels with dilations (1,3,5)."""
    for j, dil in enumerate((1, 3, 5)):
        a1 = tf.constant(SD[f"{prefix}.alpha1.{j}"].transpose(0, 2, 1))
        a2 = tf.constant(SD[f"{prefix}.alpha2.{j}"].transpose(0, 2, 1))
        xt = adain(x, style, prefix, f"adain1.{j}")
        xt = xt + (1.0 / a1) * tf.sin(a1 * xt) ** 2
        xt = conv1d(xt, f"{prefix}.convs1.{j}", dilation=dil)
        xt = adain(xt, style, prefix, f"adain2.{j}")
        xt = xt + (1.0 / a2) * tf.sin(a2 * xt) ** 2
        xt = conv1d(xt, f"{prefix}.convs2.{j}")
        x = x + xt
    return x


def generator(x, style, har):
    """ISTFTNet generator: x[1,2T,256], har[1,120T+1,22] -> wav[600T]."""
    g = "kmodel.decoder.generator"
    # stage 0
    x = tf.nn.leaky_relu(x, 0.1)
    src = strided_conv(har, f"{g}.noise_convs.0", stride=6, pad=3)  # [1,20T,128]
    src = snake_resblock(src, style, f"{g}.noise_res.0")
    x = convt(x, f"{g}.ups.0", k=20, stride=10)  # [1,20T,128]
    x = x + src
    xs = snake_resblock(x, style, f"{g}.resblocks.0")
    xs += snake_resblock(x, style, f"{g}.resblocks.1")
    x = xs / 2.0
    # stage 1
    x = tf.nn.leaky_relu(x, 0.1)
    src = conv1d(har, f"{g}.noise_convs.1")  # k1 s1 [1,120T+1,64]
    src = snake_resblock(src, style, f"{g}.noise_res.1")
    x = convt(x, f"{g}.ups.1", k=12, stride=6)  # [1,120T,64]
    x = tf.concat([x[:, 1:2], x], axis=1)  # ReflectionPad1d((1,0))
    x = x + src
    xs = snake_resblock(x, style, f"{g}.resblocks.2")
    xs += snake_resblock(x, style, f"{g}.resblocks.3")
    x = xs / 2.0
    x = tf.nn.leaky_relu(x, 0.01)
    x = conv1d(x, f"{g}.conv_post")  # [1,120T+1,22]
    spec = tf.exp(x[..., :N_FFT // 2 + 1])
    phase = tf.sin(x[..., N_FFT // 2 + 1:])
    # CustomSTFT.inverse: convT(spec*cos, Wr) - convT(spec*sin, Wi), crop n_fft/2
    wbr = tf.constant(SD[f"{g}.stft.weight_backward_real"].transpose(2, 1, 0))
    wbi = tf.constant(SD[f"{g}.stft.weight_backward_imag"].transpose(2, 1, 0))
    t = tf.shape(spec)[1]
    out_len = (t - 1) * HOP + N_FFT
    re = tf.nn.conv1d_transpose(spec * tf.cos(phase), wbr, output_shape=[1, out_len, 1],
                                strides=HOP, padding="VALID")
    im = tf.nn.conv1d_transpose(spec * tf.sin(phase), wbi, output_shape=[1, out_len, 1],
                                strides=HOP, padding="VALID")
    wav = (re - im)[:, N_FFT // 2:-(N_FFT // 2), 0]
    return wav  # [1, 600T]


def decoder(asr, f0_pred, n_pred, style_ref, har):
    """asr[1,T,128], f0/n[1,2T], style[1,128], har[1,120T+1,22] -> wav[1,600T]."""
    p = "kmodel.decoder"
    f0 = strided_conv(f0_pred[..., None], f"{p}.F0_conv", stride=2, pad=1)  # [1,T,1]
    nn_ = strided_conv(n_pred[..., None], f"{p}.N_conv", stride=2, pad=1)
    x = tf.concat([asr, f0, nn_], axis=-1)  # [1,T,130]
    x = adain_resblk(x, style_ref, f"{p}.encode")
    asr_res = conv1d(asr, f"{p}.asr_res.0")  # [1,T,64]
    for i in range(4):
        x = tf.concat([x, asr_res, f0, nn_], axis=-1)  # [1,T,322]
        if i < 3:
            x = adain_resblk(x, style_ref, f"{p}.decode.{i}")
        else:
            x = adain_resblk(x, style_ref, f"{p}.decode.3", upsample=True)  # [1,2T,·]
    return generator(x, style_ref, har)
