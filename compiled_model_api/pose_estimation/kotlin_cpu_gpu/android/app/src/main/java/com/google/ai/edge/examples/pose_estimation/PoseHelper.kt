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
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.scale
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

/**
 * Runs lightweight-OpenPose single-person pose estimation with the LiteRT CompiledModel API.
 *
 * Model: pose_256_fp16.tflite
 *  - input : 1 x 256 x 256 x 3 float32, RGB, normalized (px - 128) / 256 (NHWC, interleaved)
 *  - output: 1 x 32 x 32 x 19 float32, keypoint heatmaps (18 body keypoints + 1 background)
 *
 * Only the model (a pure-CNN backbone + refinement) runs on the GPU; the keypoint decoding
 * (argmax over each heatmap) is done here in Kotlin, so the graph stays GPU-clean (no
 * GATHER_ND, which is what makes MoveNet's baked-in decode fall back to the CPU).
 */
class PoseHelper(private val context: Context) {

  val poses: SharedFlow<PoseResult>
    get() = _poses

  private val _poses =
    MutableSharedFlow<PoseResult>(
      extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST)

  val error: SharedFlow<Throwable?>
    get() = _error

  private val _error = MutableSharedFlow<Throwable?>()

  private var detector: Detector? = null
  private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

  suspend fun initDetector(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.GPU) {
    cleanup()
    try {
      withContext(singleThreadDispatcher) {
        val model =
          CompiledModel.create(
            context.assets,
            MODEL_FILE,
            CompiledModel.Options(toAccelerator(acceleratorEnum)),
            null,
          )
        detector = Detector(model)
        Log.i(TAG, "Created a pose detector ($acceleratorEnum)")
      }
    } catch (e: Exception) {
      Log.e(TAG, "Create LiteRT from $MODEL_FILE failed: ${e.message}")
      _error.emit(e)
    }
  }

  suspend fun cleanup() {
    try {
      withContext(singleThreadDispatcher) {
        detector?.cleanup()
        detector = null
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error during cleanup: ${e.message}")
    }
  }

  suspend fun detect(bitmap: Bitmap) {
    try {
      withContext(singleThreadDispatcher) {
        detector?.detect(bitmap)?.let { if (isActive) _poses.emit(it) }
      }
    } catch (e: Exception) {
      Log.e(TAG, "Pose detect error: ${e.message}")
      _error.emit(e)
    }
  }

  private class Detector(private val model: CompiledModel) {
    private val inputBuffers = model.createInputBuffers()
    private val outputBuffers = model.createOutputBuffers()

    fun cleanup() {
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
    }

    fun detect(bitmap: Bitmap): PoseResult {
      val image = bitmap.scale(SIZE, SIZE, true)
      val input = preprocess(image)

      val start = SystemClock.uptimeMillis()
      inputBuffers[0].writeFloat(input)
      model.run(inputBuffers, outputBuffers)
      val heatmaps = outputBuffers[0].readFloat() // HM * HM * CHANNELS, NHWC
      val inferenceTime = SystemClock.uptimeMillis() - start

      return PoseResult(decode(heatmaps), bitmap.width, bitmap.height, inferenceTime)
    }

    /** Resize -> interleaved NHWC RGB float32, normalized (px - 128) / 256. */
    private fun preprocess(image: Bitmap): FloatArray {
      val numPixels = SIZE * SIZE
      val pixels = IntArray(numPixels)
      val out = FloatArray(numPixels * 3)
      image.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
      for (i in 0 until numPixels) {
        val p = pixels[i]
        val base = i * 3
        out[base + 0] = (Color.red(p) - 128f) / 256f
        out[base + 1] = (Color.green(p) - 128f) / 256f
        out[base + 2] = (Color.blue(p) - 128f) / 256f
      }
      return out
    }

    /** Argmax each keypoint heatmap -> a normalized [0,1] keypoint. */
    private fun decode(heatmaps: FloatArray): List<Keypoint> {
      val result = ArrayList<Keypoint>(NUM_KEYPOINTS)
      for (k in 0 until NUM_KEYPOINTS) {
        var best = Float.NEGATIVE_INFINITY
        var bx = 0
        var by = 0
        for (y in 0 until HM) {
          for (x in 0 until HM) {
            val v = heatmaps[(y * HM + x) * CHANNELS + k]
            if (v > best) {
              best = v
              bx = x
              by = y
            }
          }
        }
        result.add(Keypoint((bx + 0.5f) / HM, (by + 0.5f) / HM, best))
      }
      return result
    }
  }

  data class PoseResult(
    val keypoints: List<Keypoint>,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inferenceTime: Long,
  )

  enum class AcceleratorEnum {
    CPU,
    GPU,
  }

  private companion object {
    const val TAG = "PoseEstimation"
    const val MODEL_FILE = "pose_256_fp16.tflite"
    const val SIZE = 256
    const val HM = 32 // heatmap spatial size (input stride 8)
    const val CHANNELS = 19 // 18 keypoints + background
    const val NUM_KEYPOINTS = 18

    fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator =
      when (acceleratorEnum) {
        AcceleratorEnum.CPU -> Accelerator.CPU
        AcceleratorEnum.GPU -> Accelerator.GPU
      }
  }
}
