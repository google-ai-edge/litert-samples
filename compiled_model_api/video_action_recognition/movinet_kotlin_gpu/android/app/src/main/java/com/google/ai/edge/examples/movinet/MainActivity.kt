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

package com.google.ai.edge.examples.movinet

import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/**
 * Runs MoViNet-A0 streaming action recognition over a bundled reference clip
 * (13 frames of jumping jacks) and prints the running top-5 Kinetics-600
 * prediction after each frame — a deterministic, self-contained demo of the
 * streaming pipeline. The model is loaded from the app's filesDir; push it there
 * first with install_to_device.sh (it is not bundled in the APK).
 */
class MainActivity : AppCompatActivity() {

  private val executor = Executors.newSingleThreadExecutor()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val text = TextView(this).apply {
      textSize = 15f
      setPadding(36, 56, 36, 36)
      typeface = android.graphics.Typeface.MONOSPACE
    }
    setContentView(ScrollView(this).apply { addView(text) })

    fun log(s: String) = runOnUiThread { text.append(s + "\n") }

    executor.execute {
      val modelFile = File(filesDir, "movinet_a0_stream.tflite")
      if (!modelFile.exists()) {
        log("Model not found at:\n${modelFile.absolutePath}\n")
        log("Push it first:\n  ./install_to_device.sh <dir-with-tflite>\n")
        log("(build with ../conversion or download from\n litert-community/MoViNet-A0-Stream-LiteRT)")
        return@execute
      }
      val labels = assets.open("kinetics600_labels.txt").bufferedReader().readLines()
      val frames = readFrames()  // list of NCHW [3*172*172] float arrays
      log("MoViNet-A0 streaming  ·  ${frames.size} frames  ·  CompiledModel GPU\n")

      MoViNet(modelFile.absolutePath).use { model ->
        model.reset()
        var last = FloatArray(0)
        frames.forEachIndexed { t, frame ->
          val logits = model.classify(frame)
          last = logits
          val top1 = logits.indices.maxByOrNull { logits[it] }!!
          log("frame %2d  ->  %s".format(t, labels[top1]))
        }
        log("\nFinal top-5:")
        for (p in topK(last, 5)) log("  %-28s %4.1f%%".format(labels[p.first], p.second * 100))
      }
    }
  }

  /** Read the bundled clip: [N][3*172*172] NCHW float32, RGB, 0..1. */
  private fun readFrames(): List<FloatArray> {
    val bytes = assets.open("jumpingjack_frames.bin").use { it.readBytes() }
    val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val len = 3 * MoViNet.INPUT_SIZE * MoViNet.INPUT_SIZE
    return (0 until fb.limit() / len).map { i ->
      val a = FloatArray(len); fb.position(i * len); fb.get(a, 0, len); a
    }
  }

  private fun topK(logits: FloatArray, k: Int): List<Pair<Int, Float>> {
    if (logits.isEmpty()) return emptyList()
    val idx = logits.indices.sortedByDescending { logits[it] }.take(k)
    val mx = logits[idx.first()]
    var sum = 0.0
    for (v in logits) sum += Math.exp((v - mx).toDouble())
    return idx.map { it to (Math.exp((logits[it] - mx).toDouble()) / sum).toFloat() }
  }

  override fun onDestroy() {
    super.onDestroy()
    executor.shutdown()
  }
}
