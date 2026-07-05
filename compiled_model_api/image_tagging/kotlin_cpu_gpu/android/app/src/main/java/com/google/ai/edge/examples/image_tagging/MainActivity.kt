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

package com.google.ai.edge.examples.image_tagging

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
 * RAM++ multi-label image tagging. Pick a photo (or use the bundled sample) and get its tags. The
 * Swin encoder stages 0-2 and the Query2Label tag head run on the CompiledModel GPU; the fp16-fragile
 * deep Swin block and the 479 MB frozen tag bank run on CPU.
 */
class MainActivity : Activity() {

  private val bg = Executors.newSingleThreadExecutor()
  private var tagger: RamTagger? = null
  private lateinit var status: TextView
  private lateinit var tagsView: TextView
  private lateinit var imageView: ImageView
  private var bitmap: Bitmap? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(36, 90, 36, 36) }
    status = TextView(this).apply { textSize = 15f; text = "Loading RAM++ …" }
    val pick = Button(this).apply { text = "🖼  Pick image"; setOnClickListener { pickImage() } }
    imageView = ImageView(this).apply { adjustViewBounds = true }
    tagsView = TextView(this).apply { textSize = 15f; setPadding(0, 20, 0, 0) }
    root.addView(status); root.addView(pick); root.addView(imageView); root.addView(tagsView)
    setContentView(ScrollView(this).apply { addView(root) })

    bg.execute {
      try {
        tagger = RamTagger(this)
        bitmap = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
        runOnUiThread { imageView.setImageBitmap(bitmap); status.text = "Ready — tagging sample…" }
        runTag()
      } catch (e: Throwable) {
        Log.e("ImageTagging", "load", e)
        runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
      }
    }
  }

  private fun pickImage() {
    startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
      addCategory(Intent.CATEGORY_OPENABLE); type = "image/*" }, 1)
  }

  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    val uri: Uri = data?.data ?: return
    if (resultCode != RESULT_OK) return
    contentResolver.openInputStream(uri).use { bitmap = BitmapFactory.decodeStream(it) }
    imageView.setImageBitmap(bitmap)
    runTag()
  }

  private fun runTag() {
    val t = tagger ?: return; val bm = bitmap ?: return
    runOnUiThread { status.text = "Tagging on GPU…"; tagsView.text = "" }
    bg.execute {
      try {
        val t0 = System.nanoTime()
        val res = t.tag(bm)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i("ImageTagging", "TAGS ($ms ms): " + res.joinToString(" · ") { it.name })
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
          status.text = "✓ ${res.size} tags in ${ms}ms · RAM++ hybrid GPU/CPU"
          tagsView.text = res.joinToString("\n") { "• ${it.name}   %.2f".format(it.prob) }
        }
      } catch (e: Throwable) {
        Log.e("ImageTagging", "tag", e)
        runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "Failed: ${e.message}" }
      }
    }
  }

  override fun onDestroy() { super.onDestroy(); bg.shutdown(); tagger?.close() }
}
