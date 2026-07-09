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

package com.google.ai.edge.examples.yolact

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [YolactSegmenter] and exposes a single [UiState] for the screen. The YOLACT model is
 * 125 MB and reuses native buffers, so both model creation and every segment run on one confined
 * worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "yolact.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var segmenter: YolactSegmenter? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and segment a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        _uiState.update {
          it.copy(
            errorMessage =
              "Model not found. Push it first with install_to_device.sh:\n" +
                modelFile.absolutePath
          )
        }
        return@launch
      }
      try {
        segmenter = YolactSegmenter(context, modelFile.absolutePath)
        segment(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Runs instance segmentation on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        segment(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(isProcessing = false, errorMessage = t.message ?: "Segmentation failed")
        }
      }
    }
  }

  private fun segment(source: Bitmap) {
    val segmenter = segmenter ?: return
    val (instances, elapsedMs) = segmenter.segment(source)
    val result = render(source, instances)
    val labels = instances.joinToString(", ") { CocoLabels.NAMES[it.cls] }
    val detail = "${instances.size} instances" + if (labels.isNotEmpty()) ": $labels" else ""
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = result,
        resultText = detail,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Draw masks (scaled from 550) + boxes + labels onto a copy of the input. */
  private fun render(image: Bitmap, insts: List<Instance>): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val S = YolactSegmenter.SIZE
    val sx = out.width.toFloat() / S
    val sy = out.height.toFloat() / S
    val mp = Paint()
    val bp = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = out.width / 200f
    }
    val tp = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.WHITE
      textSize = out.width / 28f
      setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    // masks: composite a translucent color per instance
    val row = IntArray(out.width)
    for (ins in insts) {
      val c = (0x88 shl 24) or (Palette.color(ins.cls) and 0x00FFFFFF)
      mp.color = c
      for (yy in 0 until out.height) {
        val my = (yy / sy).toInt().coerceIn(0, S - 1)
        var started = -1
        for (xx in 0 until out.width) {
          val on = ins.mask[my * S + (xx / sx).toInt().coerceIn(0, S - 1)]
          if (on && started < 0) {
            started = xx
          }
          if ((!on || xx == out.width - 1) && started >= 0) {
            canvas.drawRect(started.toFloat(), yy.toFloat(), xx.toFloat(), (yy + 1).toFloat(), mp)
            started = -1
          }
        }
      }
    }
    for (ins in insts) {
      bp.color = Palette.color(ins.cls)
      canvas.drawRect(ins.x1 * S * sx, ins.y1 * S * sy, ins.x2 * S * sx, ins.y2 * S * sy, bp)
      canvas.drawText("${CocoLabels.NAMES[ins.cls]} ${(ins.score * 100).toInt()}%",
        ins.x1 * S * sx + 6, ins.y1 * S * sy + tp.textSize, tp)
    }
    return out
  }

  override fun onCleared() {
    super.onCleared()
    segmenter?.close()
  }
}
