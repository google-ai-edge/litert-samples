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

package com.google.ai.edge.examples.face_restoration

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.exifinterface.media.ExifInterface
import java.util.concurrent.Executors

private const val TAG = "FaceRestoration"
private const val MAX_INPUT_SIZE = 1024

/**
 * Pick a photo -> detect the largest face (YuNet) -> FFHQ-align to 512 -> restore (GFPGAN) ->
 * before/after slider. All models run on the LiteRT CompiledModel GPU.
 */
class MainActivity : ComponentActivity() {

  private lateinit var compareView: CompareView
  private lateinit var statusText: TextView
  private lateinit var pickButton: Button

  private var restorer: FaceRestorer? = null
  private var detector: FaceDetector? = null
  private val executor = Executors.newSingleThreadExecutor()
  private var isProcessing = false

  private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
    uri?.let { processImage(it) }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    val root = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(0, 48, 0, 0)
    }
    statusText = TextView(this).apply {
        textSize = 16f
        setPadding(24, 8, 24, 8)
        text = "Loading model..."
    }
    root.addView(statusText)
    pickButton = Button(this).apply {
      text = "Select Face Photo"
      isEnabled = false
      setOnClickListener { imagePicker.launch("image/*") }
    }
    root.addView(
      pickButton,
      LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        .apply { gravity = Gravity.CENTER_HORIZONTAL },
    )
    compareView = CompareView(this).apply {
      layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
    }
    root.addView(compareView)
    setContentView(root)

    executor.execute {
      try {
        restorer = FaceRestorer(this)
        detector = try { FaceDetector(this) } catch (e: Exception) {
          Log.w(TAG, "Face detector unavailable, using center crop", e)
          null
        }
        runOnUiThread {
            statusText.text = "Ready — select a face photo"
            pickButton.isEnabled = true
        }
      } catch (e: Exception) {
        Log.e(TAG, "Model load failed", e)
        runOnUiThread { statusText.text = "Failed: ${e.message}" }
      }
    }
  }

  private fun processImage(uri: Uri) {
    val r = restorer ?: return
    if (isProcessing) return
    isProcessing = true
    pickButton.isEnabled = false
    statusText.text = "Restoring..."
    executor.execute {
      try {
        val bitmap = loadBitmap(uri) ?: throw Exception("Failed to load image")
        val before = prepareAligned(bitmap)
        bitmap.recycle()
        val t = System.nanoTime()
        val after = r.restore(before)
        val ms = (System.nanoTime() - t) / 1_000_000
        runOnUiThread {
          compareView.setImages(before, after)
          statusText.text = "Restored 512x512 in ${ms}ms — drag to compare"
          pickButton.isEnabled = true
        }
      } catch (e: Exception) {
        Log.e(TAG, "Restore failed", e)
        runOnUiThread {
            statusText.text = "Error: ${e.message}"
            pickButton.isEnabled = true
        }
      }
      isProcessing = false
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

  private fun loadBitmap(uri: Uri): Bitmap? {
    val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
    var sampleSize = 1
    while (opts.outWidth / sampleSize > MAX_INPUT_SIZE || opts.outHeight / sampleSize > MAX_INPUT_SIZE) sampleSize *= 2
    val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
    val bitmap = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, decodeOpts) } ?: return null
    val rotation = contentResolver.openInputStream(uri)?.use {
      when (ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90f
        ExifInterface.ORIENTATION_ROTATE_180 -> 180f
        ExifInterface.ORIENTATION_ROTATE_270 -> 270f
        else -> 0f
      }
    } ?: 0f
    if (rotation == 0f) return bitmap
    val m = Matrix().apply { postRotate(rotation) }
    val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    bitmap.recycle()
    return rotated
  }

  override fun onDestroy() {
    super.onDestroy()
    restorer?.close()
    detector?.close()
    executor.shutdown()
  }
}
