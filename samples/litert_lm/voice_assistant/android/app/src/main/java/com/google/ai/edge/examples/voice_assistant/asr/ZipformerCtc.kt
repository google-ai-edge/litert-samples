/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

// Adapted from this repository's samples/litert_model_zoo (models/zipformer/ZipformerAsr.kt),
// by way of john-rocky/hfmodels-android (commit 3086d647):
// litert/src/main/kotlin/io/github/johnrocky/hfmodels/litert/ZipformerCtc.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.asr

import java.io.File

/**
 * The `zipformer_ctc` graph contract (litert-community/Zipformer-medium-CR-CTC-LiteRT, every
 * size): inputs are the log(1e-10)-padded fbank `[1, frames, 80]` plus four additive attention
 * biases (0 = real frame, -1000 = padding), one per internal frame rate (`[1, t50]`, `[1, t50/2]`,
 * `[1, t50/4]`, `[1, t50/8]`, rounded up); the output is raw CTC logits `[1, tOut, classes]` at 25
 * Hz. For the published 16 s window: frames 1600, biases 796 / 398 / 199 / 100, logits 398 x 500.
 * Inputs are found by their size, as this repository's model zoo app does
 * (samples/litert_model_zoo/android, ZipformerAsr.kt), so the converter's tensor order and names
 * do not matter.
 */
internal class ZipformerCtc(val frames: Int, val blank: Int) {
  /** Frames after the 2x subsampling (50 Hz). */
  val t50 = (frames - 7) / 2

  /** Bias lengths at 50, 25, 12.5 and 6.25 Hz: t50 / ds, rounded up. */
  val biasLengths = intArrayOf(1, 2, 4, 8).map { ds -> (t50 + ds - 1) / ds }

  /** Output frames (25 Hz). */
  val tOut = biasLengths[1]

  /**
   * Additive bias for internal rate [r] (0..3): 0 for the frames that carry audio, -1000 for
   * padding.
   */
  fun bias(r: Int, valid50: Int): FloatArray {
    val len = biasLengths[r]
    // 1, 2, 4, 8
    val ds = t50 / len + if (t50 % len != 0) 1 else 0
    return FloatArray(len) { i -> if (i * ds < valid50) 0f else -1000f }
  }

  /** 50 Hz frames that carry audio for [fbankFrames] real fbank frames. */
  fun valid50(fbankFrames: Int): Int = (minOf(fbankFrames, frames) - 7) / 2

  /** 25 Hz output frames to decode. */
  fun validOut(valid50: Int): Int = minOf((valid50 + 1) / 2, tOut)

  companion object {
    /**
     * `tokens.txt`: one `<piece> <id>` per line (the zoo app's reading: the id after the last
     * space).
     */
    fun readTokens(file: File): Map<Int, String> =
      file
        .readLines(Charsets.UTF_8)
        .filter { it.isNotBlank() }
        .associate { line ->
          val cut = line.lastIndexOf(' ')
          line.substring(cut + 1).trim().toInt() to line.substring(0, cut)
        }

    /**
     * Greedy CTC over the real frames: argmax per frame, drop blanks and repeats, `▁` -> space,
     * trim.
     */
    fun decode(
      logits: FloatArray,
      validOut: Int,
      classes: Int,
      blank: Int,
      pieces: Map<Int, String>,
    ): String {
      val sb = StringBuilder()
      var prev = -1
      for (t in 0 until validOut) {
        var best = -Float.MAX_VALUE
        var arg = 0
        val base = t * classes
        for (c in 0 until classes) {
          val v = logits[base + c]
          if (v > best) {
            best = v
            arg = c
          }
        }
        if (arg != blank && arg != prev) {
          sb.append(pieces[arg] ?: "")
        }
        prev = arg
      }
      return sb.toString().replace('▁', ' ').trim()
    }
  }
}
