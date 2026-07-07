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

package com.google.ai.edge.examples.sound_event_detection

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * PANNs CNN14 audio tagging (AudioSet, 527 sound-event classes), on-device, with the CNN running
 * fully on the LiteRT CompiledModel GPU (ML Drift).
 *
 *   waveform[320000] --[Kotlin log-mel]--> logmel[1,1,1001,64] --[GPU CNN14]--> probs[527] (sigmoid)
 *
 * Only the log-mel front-end is host-side (see MelSpectrogram for why: the power spectrum overflows
 * fp16). The whole CNN14 body — bn0 + 6 conv blocks + pooling + 2 FC + sigmoid — is a pure CNN and
 * rides the GPU graph (corr 1.0 vs PyTorch in fp16). AudioSet tagging is multi-label, so the output
 * is per-class probabilities, not a softmax: several tags can be high at once.
 *
 * The tflite is loaded from filesDir (162 MB, pushed via install_to_device.sh).
 */
class AudioTagger(private val context: Context) : Closeable {

  companion object {
    const val MODEL = "cnn14_audioset_fp16.tflite"
    const val SAMPLES = MelSpectrogram.CLIP_SAMPLES   // 320000 (10 s @ 32 kHz)
    const val NUM_CLASSES = 527
  }

  private val labels: Array<String> = context.assets.open("audioset_labels.txt").bufferedReader()
    .useLines { lines -> lines.filter { it.isNotBlank() }.toList() }.toTypedArray()

  private val mel = MelSpectrogram(context)

  private val model: CompiledModel = run {
    val f = File(context.filesDir, MODEL)
    check(f.exists()) { "Model not found: $MODEL. Push it first:\n  ./install_to_device.sh" }
    CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
  }
  private val inBuf = model.createInputBuffers()
  private val outBuf = model.createOutputBuffers()

  data class Tag(val label: String, val prob: Float)
  data class Result(val tags: List<Tag>, val probs: FloatArray, val melMs: Long, val gpuMs: Long)

  init { require(labels.size == NUM_CLASSES) { "labels=${labels.size}, expected $NUM_CLASSES" } }

  /** Tag a 32 kHz mono clip (pads/truncates to 10 s). Returns the top-K tags by probability. */
  fun tag(audio: FloatArray, topK: Int = 10): Result {
    val t0 = System.nanoTime()
    val logmel = mel.compute(audio)                 // [1001*64]
    val t1 = System.nanoTime()
    inBuf[0].writeFloat(logmel)
    model.run(inBuf, outBuf)
    val probs = outBuf[0].readFloat()               // [527]
    val t2 = System.nanoTime()

    val idx = (0 until NUM_CLASSES).sortedByDescending { probs[it] }.take(topK)
    val tags = idx.map { Tag(labels[it], probs[it]) }
    return Result(tags, probs, (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000)
  }

  override fun close() {
    inBuf.forEach { it.close() }
    outBuf.forEach { it.close() }
    model.close()
  }
}
