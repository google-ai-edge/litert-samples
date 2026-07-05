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

package com.google.ai.edge.examples.image_quality_assessment

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
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
 * NIMA image quality assessment. Pick a photo (or use the bundled sample) and get its aesthetic and
 * technical quality scores (1-10). Both MobileNet models run on the LiteRT CompiledModel GPU.
 */
class MainActivity : Activity() {

  private val bg = Executors.newSingleThreadExecutor()
  private var scorer: NimaScorer? = null
  private lateinit var status: TextView
  private lateinit var scoreView: TextView
  private lateinit var imageView: ImageView
  private var bitmap: Bitmap? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(40, 100, 40, 40) }
    status = TextView(this).apply { textSize = 15f; text = "Loading NIMA…" }
    val pick = Button(this).apply { text = "🖼  Pick image"; setOnClickListener { pickImage() } }
    imageView = ImageView(this).apply { adjustViewBounds = true }
    scoreView = TextView(this).apply { textSize = 22f; setPadding(0, 24, 0, 0) }
    root.addView(status); root.addView(pick); root.addView(imageView); root.addView(scoreView)
    setContentView(ScrollView(this).apply { addView(root) })

    bg.execute {
      try {
        scorer = NimaScorer(this)
        bitmap = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
        runOnUiThread { imageView.setImageBitmap(bitmap); status.text = "Ready — scoring sample…" }
        runScore()
      } catch (e: Throwable) {
        Log.e("NIMA", "load", e)
        runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
      }
    }
  }

  private fun pickImage() = startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
    addCategory(Intent.CATEGORY_OPENABLE); type = "image/*" }, 1)

  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    val uri: Uri = data?.data ?: return
    if (resultCode != RESULT_OK) return
    contentResolver.openInputStream(uri).use { bitmap = BitmapFactory.decodeStream(it) }
    imageView.setImageBitmap(bitmap)
    runScore()
  }

  private fun runScore() {
    val sc = scorer ?: return; val bm = bitmap ?: return
    runOnUiThread { status.text = "Scoring on GPU…" }
    bg.execute {
      try {
        val t0 = System.nanoTime()
        val r = sc.score(bm)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i("NIMA", "SCORES aesthetic=%.3f technical=%.3f ms=%d".format(r.aesthetic, r.technical, ms))
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
          status.text = "✓ scored in ${ms}ms · NIMA MobileNet, CompiledModel GPU"
          scoreView.text = "Aesthetic  %.2f / 10\nTechnical  %.2f / 10".format(r.aesthetic, r.technical)
        }
      } catch (e: Throwable) {
        Log.e("NIMA", "score", e)
        runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "Failed: ${e.message}" }
      }
    }
  }

  override fun onDestroy() { super.onDestroy(); bg.shutdown(); scorer?.close() }
}
