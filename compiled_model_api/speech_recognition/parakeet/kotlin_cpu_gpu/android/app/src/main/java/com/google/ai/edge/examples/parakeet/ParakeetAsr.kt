/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.parakeet

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * NVIDIA Parakeet (FastConformer-CTC, parakeet-tdt_ctc-110m, the CTC head) speech recognition, fully on
 * the LiteRT CompiledModel GPU (ML Drift / LITERT_CL) on a Pixel 8a.
 *
 * Pipeline:  host log-mel  ->  [GPU] 17-layer FastConformer encoder + CTC Conv1d  ->  host greedy-CTC +
 * SentencePiece detokenize. The model is a single GPU graph with a FIXED ~16 s window: inputs are the
 * zero-padded mel [1,80,1601] and a frame mask [1,201] (1 = real, 0 = pad); the encoder masking is folded
 * in as a GPU-clean additive attention bias + conv frame-mask, so any audio up to 16 s is padded to the
 * window and decoded over its real frames only.
 *
 * Device-verified (Pixel 8a): real-frame logits corr 0.99997 vs PyTorch, transcript exact, 3105/3105 ops
 * on LITERT_CL (1 partition). ~330 ms GPU + ~70 ms host mel ≈ 0.4 s per 16 s window (the async run()
 * enqueue returns in ~40 ms; the GPU compute completes at output readback).
 */
class ParakeetAsr(private val ctx: Context) : Closeable {

  companion object {
    const val MODEL = "parakeet_ship_fp16.tflite"
    const val MEL_LEN = 1601 // ~16 s window (mel frames)
    const val T_PRIME = 201 // subsampled length of MEL_LEN (factor 8)
    const val N_MEL = MelSpectrogram.N_MEL
    const val VOCAB = 1024 // SentencePiece pieces; CTC blank id = VOCAB (1024)
    const val BLANK = VOCAB
    const val NCLASS = VOCAB + 1 // 1025 CTC logits
    const val MAX_SAMPLES = (MEL_LEN - 1) * MelSpectrogram.HOP // 256000 = 16 s @ 16 kHz
  }

  private fun loadModel(): CompiledModel {
    val f = File(ctx.filesDir, MODEL)
    check(f.exists()) { "Model not found: $MODEL. Push it first: ./install_to_device.sh" }
    return CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
  }

  private val model = loadModel()
  private val inBufs = model.createInputBuffers()
  private val outBufs = model.createOutputBuffers()
  // resolve slots by float capacity (robust to converter ordering)
  private val melSlot = inBufs.indexOfFirst { it.readFloat().size == N_MEL * MEL_LEN }
  private val maskSlot = inBufs.indexOfFirst { it.readFloat().size == T_PRIME }
  private val logitsSlot = outBufs.indexOfFirst { it.readFloat().size == T_PRIME * NCLASS }

  private val mel = MelSpectrogram(loadFb())
  private val pieces: List<String> = ctx.assets.open("tokens.txt").bufferedReader().readLines()

  private val melInput = FloatArray(N_MEL * MEL_LEN)
  private val maskInput = FloatArray(T_PRIME)

  private fun loadFb(): FloatArray {
    val raw = ctx.assets.open("mel_fb.bin").readBytes()
    val fb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    return FloatArray(N_MEL * MelSpectrogram.N_BIN).also { fb.get(it) }
  }

  data class Result(val text: String, val melMs: Long, val gpuMs: Long)

  /** @param audio mono PCM in [-1,1] at 16 kHz (clipped to the 16 s window). */
  fun transcribe(audio: FloatArray): Result {
    val clip = if (audio.size > MAX_SAMPLES) audio.copyOf(MAX_SAMPLES) else audio

    val t0 = System.nanoTime()
    val r = mel.compute(clip)
    val nfr = minOf(r.frames, MEL_LEN)
    // pack into the fixed window [N_MEL, MEL_LEN], zero-padded
    java.util.Arrays.fill(melInput, 0f)
    for (m in 0 until N_MEL) {
      System.arraycopy(r.logmel, m * r.frames, melInput, m * MEL_LEN, nfr)
    }
    val tr = minOf(MelSpectrogram.subLen(nfr), T_PRIME)
    for (i in 0 until T_PRIME) maskInput[i] = if (i < tr) 1f else 0f
    val t1 = System.nanoTime()

    inBufs[melSlot].writeFloat(melInput)
    inBufs[maskSlot].writeFloat(maskInput)
    model.run(inBufs, outBufs)
    val logits = outBufs[logitsSlot].readFloat() // [T_PRIME * NCLASS]
    val t2 = System.nanoTime()

    return Result(decode(logits, tr), (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000)
  }

  /** Greedy CTC over the real frames: argmax per frame, drop blanks and consecutive repeats, detok. */
  private fun decode(logits: FloatArray, tr: Int): String {
    val ids = ArrayList<Int>()
    var prev = -1
    for (t in 0 until tr) {
      var best = -Float.MAX_VALUE
      var arg = 0
      val base = t * NCLASS
      for (c in 0 until NCLASS) {
        val v = logits[base + c]
        if (v > best) {
          best = v
          arg = c
        }
      }
      if (arg != BLANK && arg != prev) ids.add(arg)
      prev = arg
    }
    val sb = StringBuilder()
    for (id in ids) if (id < pieces.size) sb.append(pieces[id])
    return sb.toString().replace('▁', ' ').trim()
  }

  override fun close() {
    inBufs.forEach { it.close() }
    outBufs.forEach { it.close() }
    model.close()
  }
}
