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

package com.google.ai.edge.examples.movinet

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [MoViNet] streaming model and exposes a single [UiState]. On startup it loads the GPU
 * compiled model from filesDir, the Kinetics-600 labels, and the bundled reference clip, then
 * auto-streams the clip once. Each [run] loops the 13 frames, calling [MoViNet.classify] per frame
 * and STREAMING the running "frame N -> label" log into [UiState.outputText] after every frame so
 * the predictions fill in live. The model reuses native buffers, so all model calls run on one
 * confined worker.
 *
 * The 15 MB model is NOT bundled — it is pushed to the app's filesDir by `install_to_device.sh`;
 * until then the status line asks the user to run it.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "movinet_a0_stream.tflite"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var model: MoViNet? = null
  private var labels: List<String> = emptyList()
  private var frames: List<FloatArray> = emptyList()

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState(statusMessage = "Loading MoViNet-A0 (GPU)…"))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val modelFile = File(context.filesDir, MODEL_FILE)
        if (!modelFile.exists()) {
          _uiState.update {
            it.copy(
              errorMessage =
                "Model not found at:\n${modelFile.absolutePath}\n\n" +
                  "Push it first:\n  ./install_to_device.sh <dir-with-tflite>\n\n" +
                  "(build with ../conversion or download from\n" +
                  " litert-community/MoViNet-A0-Stream-LiteRT)"
            )
          }
          return@launch
        }
        labels = context.assets.open("kinetics600_labels.txt").bufferedReader().readLines()
        frames = readFrames() // list of NCHW [3*172*172] float arrays
        model = MoViNet(modelFile.absolutePath)
        _uiState.update {
          it.copy(
            isModelReady = true,
            statusMessage = "MoViNet-A0 streaming  ·  ${frames.size} frames  ·  CompiledModel GPU",
          )
        }
        stream()
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Streams the bundled clip through the model, appending "frame N -> label" per frame. */
  fun run() {
    if (model == null || frames.isEmpty()) return
    viewModelScope.launch(inferenceDispatcher) { stream() }
  }

  private fun stream() {
    val model = model ?: return
    _uiState.update {
      it.copy(isRunning = true, outputText = "", statusMessage = "Streaming…", errorMessage = null)
    }
    model.reset()
    val sb = StringBuilder()
    var last = FloatArray(0)
    frames.forEachIndexed { t, frame ->
      val logits = model.classify(frame)
      last = logits
      val top1 = logits.indices.maxByOrNull { logits[it] }!!
      sb.append("frame %2d  ->  %s".format(t, labels[top1])).append('\n')
      _uiState.update { it.copy(outputText = sb.toString()) }
    }
    sb.append("\nFinal top-5:\n")
    for (p in topK(last, 5)) {
      sb.append("  %-28s %4.1f%%".format(labels[p.first], p.second * 100)).append('\n')
    }
    _uiState.update {
      it.copy(
        isRunning = false,
        outputText = sb.toString(),
        statusMessage = "Done  ·  ${frames.size} frames",
      )
    }
  }

  /** Read the bundled clip: [N][3*172*172] NCHW float32, RGB, 0..1. */
  private fun readFrames(): List<FloatArray> {
    val bytes = context.assets.open("jumpingjack_frames.bin").use { it.readBytes() }
    val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val len = 3 * MoViNet.INPUT_SIZE * MoViNet.INPUT_SIZE
    return (0 until fb.limit() / len).map { i ->
      val a = FloatArray(len)
      fb.position(i * len)
      fb.get(a, 0, len)
      a
    }
  }

  private fun topK(logits: FloatArray, k: Int): List<Pair<Int, Float>> {
    if (logits.isEmpty()) return emptyList()
    val idx = logits.indices.sortedByDescending { logits[it] }.take(k)
    val mx = logits[idx.first()]
    var sum = 0.0
    for (v in logits) {
      sum += Math.exp((v - mx).toDouble())
    }
    return idx.map { it to (Math.exp((logits[it] - mx).toDouble()) / sum).toFloat() }
  }

  override fun onCleared() {
    super.onCleared()
    model?.close()
  }
}
