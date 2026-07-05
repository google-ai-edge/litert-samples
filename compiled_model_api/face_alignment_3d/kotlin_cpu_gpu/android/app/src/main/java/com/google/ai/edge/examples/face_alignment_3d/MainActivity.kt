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

package com.google.ai.edge.examples.face_alignment_3d

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * 3DDFA_V2 3D face alignment. Pick a frontal-face photo (or use the bundled sample) and see the 68
 * 3D face landmarks. The MobileNetV1 3DMM regressor runs on the LiteRT CompiledModel GPU; the 68
 * landmarks are reconstructed from the BFM bases host-side.
 */
class MainActivity : Activity() {

  private val bg = Executors.newSingleThreadExecutor()
  private var tddfa: TddfaLandmarks? = null
  private lateinit var status: TextView
  private lateinit var imageView: ImageView
  private var bitmap: Bitmap? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(40, 100, 40, 40) }
    status = TextView(this).apply { textSize = 15f; text = "Loading 3DDFA…" }
    val pick = Button(this).apply { text = "🖼  Pick face photo"; setOnClickListener { pickImage() } }
    imageView = ImageView(this).apply { adjustViewBounds = true }
    root.addView(status); root.addView(pick); root.addView(imageView)
    setContentView(ScrollView(this).apply { addView(root) })

    bg.execute {
      try {
        tddfa = TddfaLandmarks(this)
        bitmap = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }.copy(Bitmap.Config.ARGB_8888, true)
        runOnUiThread { imageView.setImageBitmap(bitmap); status.text = "Ready — detecting…" }
        runDetect()
      } catch (e: Throwable) {
        Log.e("FaceAlignment", "load", e)
        runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
      }
    }
  }

  private fun pickImage() = startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
    addCategory(Intent.CATEGORY_OPENABLE); type = "image/*" }, 1)

  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    val uri: Uri = data?.data ?: return
    if (resultCode != RESULT_OK) return
    contentResolver.openInputStream(uri).use { bitmap = BitmapFactory.decodeStream(it).copy(Bitmap.Config.ARGB_8888, true) }
    runDetect()
  }

  private fun runDetect() {
    val t = tddfa ?: return; val bm = bitmap ?: return
    runOnUiThread { status.text = "Detecting landmarks on GPU…" }
    bg.execute {
      try {
        val t0 = System.nanoTime()
        val lm = t.landmarks(bm)
        val ms = (System.nanoTime() - t0) / 1_000_000
        if (lm == null) {
          runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xE0, 0xB2)); status.text = "No face detected — try a frontal face photo." }
          return@execute
        }
        Log.i("FaceAlignment", "68 landmarks in ${ms}ms")
        val overlay = draw(bm, lm)
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
          status.text = "✓ 68 landmarks in ${ms}ms · 3DDFA_V2 MobileNet, CompiledModel GPU"
          imageView.setImageBitmap(overlay)
        }
      } catch (e: Throwable) {
        Log.e("FaceAlignment", "detect", e)
        runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "Failed: ${e.message}" }
      }
    }
  }

  private fun draw(bm: Bitmap, lm: FloatArray): Bitmap {
    val out = bm.copy(Bitmap.Config.ARGB_8888, true)
    val c = Canvas(out)
    val rad = (bm.width.coerceAtLeast(bm.height) / 220f).coerceAtLeast(2f)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x00, 0xE5, 0x76); style = Paint.Style.FILL }
    for (n in 0 until lm.size / 2) c.drawCircle(lm[n * 2], lm[n * 2 + 1], rad, paint)
    return out
  }

  override fun onDestroy() { super.onDestroy(); bg.shutdown(); tddfa?.close() }
}
