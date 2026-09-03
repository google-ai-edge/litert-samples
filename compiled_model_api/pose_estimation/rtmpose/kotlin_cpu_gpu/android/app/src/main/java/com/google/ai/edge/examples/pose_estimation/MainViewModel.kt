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

package com.google.ai.edge.examples.pose_estimation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
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
 * Owns the [RtmPoseEstimator] and exposes a single [UiState] for the screen. The estimator reuses
 * native input and output buffers, so model creation and every pose run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    /** COCO 17-keypoint skeleton edges. */
    private val SKELETON =
      arrayOf(
        5 to 7,
        7 to 9,
        6 to 8,
        8 to 10,
        11 to 13,
        13 to 15,
        12 to 14,
        14 to 16,
        5 to 6,
        11 to 12,
        5 to 11,
        6 to 12,
        0 to 5,
        0 to 6,
        0 to 1,
        0 to 2,
        1 to 3,
        2 to 4,
      )

    /** A keypoint is drawn, and counted as visible, only above this confidence. */
    private const val KEYPOINT_SCORE_THRESHOLD = 0.3f

    /**
     * Height in pixels of the rendered skeleton overlay. The 192x256 crop is drawn 3.75x larger so
     * that the bone and joint sizes below stay in the proportion the original demo used.
     */
    private const val OVERLAY_HEIGHT_PX = 960

    private const val BONE_STROKE_WIDTH_PX = 6f
    private const val JOINT_RADIUS_PX = 7f

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var estimator: RtmPoseEstimator? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  private val bonePaint =
    Paint().apply {
      color = Color.rgb(0, 230, 0)
      strokeWidth = BONE_STROKE_WIDTH_PX
      isAntiAlias = true
    }
  private val jointPaint =
    Paint().apply {
      color = Color.rgb(255, 40, 40)
      isAntiAlias = true
    }
  private val imagePaint = Paint().apply { isFilterBitmap = true }

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh). This demo bundles no image,
    // so it waits for the user to pick one.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, RtmPoseEstimator.MODEL)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        estimator = RtmPoseEstimator(context)
        _uiState.update { it.copy(isModelReady = true) }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Estimates the pose of the person in a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        estimate(cropPerson(context.loadOrientedBitmap(uri)))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_estimate_failed),
          )
        }
      }
    }
  }

  private fun estimate(crop: Bitmap) {
    val estimator = estimator ?: return
    val start = System.nanoTime()
    val keypoints = estimator.estimate(bitmapToRgb(crop))
    val elapsedMs = (System.nanoTime() - start) / 1_000_000
    val visible = keypoints.count { it.score > KEYPOINT_SCORE_THRESHOLD }
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = drawSkeleton(crop, keypoints),
        visibleKeypoints = visible,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Center-crops to the model's 3:4 (192x256) aspect, then resizes. */
  private fun cropPerson(source: Bitmap): Bitmap {
    val aspectRatio = RtmPoseEstimator.W.toFloat() / RtmPoseEstimator.H
    val width = source.width
    val height = source.height
    val crop =
      if (width.toFloat() / height > aspectRatio) {
        val cropWidth = (height * aspectRatio).toInt()
        Bitmap.createBitmap(source, (width - cropWidth) / 2, 0, cropWidth, height)
      } else {
        val cropHeight = (width / aspectRatio).toInt()
        Bitmap.createBitmap(source, 0, (height - cropHeight) / 2, width, cropHeight)
      }
    return Bitmap.createScaledBitmap(crop, RtmPoseEstimator.W, RtmPoseEstimator.H, true)
  }

  /** Flattens the crop to a row-major RGB float array in the [0, 255] range. */
  private fun bitmapToRgb(bitmap: Bitmap): FloatArray {
    val pixelCount = bitmap.width * bitmap.height
    val pixels = IntArray(pixelCount)
    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    val rgb = FloatArray(pixelCount * 3)
    for (i in 0 until pixelCount) {
      val pixel = pixels[i]
      rgb[i * 3] = ((pixel shr 16) and 0xFF).toFloat()
      rgb[i * 3 + 1] = ((pixel shr 8) and 0xFF).toFloat()
      rgb[i * 3 + 2] = (pixel and 0xFF).toFloat()
    }
    return rgb
  }

  /** Draws the COCO skeleton over an enlarged copy of the person crop. */
  private fun drawSkeleton(crop: Bitmap, keypoints: List<RtmPoseEstimator.Keypoint>): Bitmap {
    val scale = OVERLAY_HEIGHT_PX.toFloat() / crop.height
    val width = (crop.width * scale).toInt()
    val output = Bitmap.createBitmap(width, OVERLAY_HEIGHT_PX, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(output)
    val destination = RectF(0f, 0f, width.toFloat(), OVERLAY_HEIGHT_PX.toFloat())
    canvas.drawBitmap(crop, null, destination, imagePaint)
    for ((from, to) in SKELETON) {
      if (
        keypoints[from].score > KEYPOINT_SCORE_THRESHOLD &&
          keypoints[to].score > KEYPOINT_SCORE_THRESHOLD
      ) {
        canvas.drawLine(
          keypoints[from].x * scale,
          keypoints[from].y * scale,
          keypoints[to].x * scale,
          keypoints[to].y * scale,
          bonePaint,
        )
      }
    }
    for (keypoint in keypoints) {
      if (keypoint.score > KEYPOINT_SCORE_THRESHOLD) {
        canvas.drawCircle(keypoint.x * scale, keypoint.y * scale, JOINT_RADIUS_PX, jointPaint)
      }
    }
    return output
  }

  override fun onCleared() {
    super.onCleared()
    estimator?.close()
  }
}
