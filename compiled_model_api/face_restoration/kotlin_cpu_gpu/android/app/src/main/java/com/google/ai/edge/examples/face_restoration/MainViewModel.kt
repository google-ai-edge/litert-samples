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

package com.google.ai.edge.examples.face_restoration

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
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
 * Owns the [FaceRestorer] and [FaceDetector] and exposes a single [UiState] for the screen. Both
 * helpers reuse native input and output buffers across calls, so model creation and every model
 * run are confined to one single-threaded dispatcher.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val TAG = "FaceRestoration"
    private const val MODEL_GFPGAN = "gfpgan_fp16.tflite"
    private const val MODEL_YUNET = "yunet_fp16.tflite"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var restorer: FaceRestorer? = null
  private var detector: FaceDetector? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Both models are staged in filesDir by install_to_device.sh (too large to bundle).
    viewModelScope.launch(inferenceDispatcher) {
      val missing =
        listOf(MODEL_GFPGAN, MODEL_YUNET).map { File(context.filesDir, it) }.firstOrNull {
          !it.exists()
        }
      if (missing != null) {
        val message = context.getString(R.string.error_model_missing, missing.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        restorer = FaceRestorer(context)
        detector = FaceDetector(context)
        _uiState.update { it.copy(isModelReady = true) }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Restores a face in a gallery image picked by the user. */
  fun process(uri: Uri) {
    val restorer = restorer ?: return
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        val bitmap =
          context.loadBitmap(uri)
            ?: throw IllegalStateException(context.getString(R.string.error_load_image))
        val before = prepareAligned(bitmap)
        bitmap.recycle()
        val start = System.nanoTime()
        val after = restorer.restore(before)
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
        _uiState.update {
          it.copy(
            isProcessing = false,
            beforeImage = before,
            afterImage = after,
            inferenceTimeMs = elapsedMs,
          )
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_restore_failed),
          )
        }
      }
    }
  }

  /** Detect the largest face and FFHQ-align it to 512x512. Falls back to a center-square crop. */
  private fun prepareAligned(src: Bitmap): Bitmap {
    val d = detector
    if (d != null) {
      try {
        val sz = FaceDetector.SIZE
        val det = Bitmap.createScaledBitmap(src, sz, sz, true)
        val px = IntArray(sz * sz)
        det.getPixels(px, 0, sz, 0, 0, sz, sz)
        val rgb = FloatArray(sz * sz * 3)
        var i = 0
        for (p in px) {
          rgb[i++] = ((p shr 16) and 0xFF).toFloat()
          rgb[i++] = ((p shr 8) and 0xFF).toFloat()
          rgb[i++] = (p and 0xFF).toFloat()
        }
        det.recycle()
        val face = d.detect(rgb).maxByOrNull { it.score }
        if (face != null) {
          val sx = src.width.toFloat() / sz
          val sy = src.height.toFloat() / sz
          val lm = FloatArray(10)
          for (j in 0 until 5) {
            lm[2 * j] = face.landmarks[2 * j] * sx
            lm[2 * j + 1] = face.landmarks[2 * j + 1] * sy
          }
          Log.i(TAG, "Face detected (score ${"%.2f".format(face.score)}), FFHQ-aligned")
          return FaceAligner.align(src, lm)
        }
        Log.w(TAG, "No face detected — center crop")
      } catch (e: Exception) {
        Log.w(TAG, "Alignment failed — center crop", e)
      }
    }
    return restorer!!.toFaceInput(centerSquareCrop(src))
  }

  private fun centerSquareCrop(src: Bitmap): Bitmap {
    val s = minOf(src.width, src.height)
    return Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
  }

  override fun onCleared() {
    super.onCleared()
    restorer?.close()
    detector?.close()
  }
}
