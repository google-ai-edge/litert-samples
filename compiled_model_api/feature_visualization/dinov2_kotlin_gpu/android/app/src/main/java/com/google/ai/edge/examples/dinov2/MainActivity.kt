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

package com.google.ai.edge.examples.dinov2

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
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

private const val TAG = "DINOv2"
private const val PICK_IMAGE = 1

/**
 * DINOv2 dense-feature visualization demo. Pick a photo (or use the bundled
 * sample) to see the top-3 PCA of the DINOv2 patch features as an RGB overlay,
 * side by side with the image. Semantically similar regions (object parts vs
 * background) share a color. The DINOv2 ViT-S/14 backbone runs on the LiteRT
 * CompiledModel GPU, and the PCA is host-side.
 */
class MainActivity : Activity() {

    private val background = Executors.newSingleThreadExecutor()
    private var extractor: Dinov2Features? = null
    private var bitmap: Bitmap? = null

    private lateinit var status: TextView
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 100, 40, 40)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading DINOv2…"
        }
        val pick = Button(this).apply {
            text = "🖼  Pick image"
            setOnClickListener { pickImage() }
        }
        imageView = ImageView(this).apply { adjustViewBounds = true }
        root.addView(status)
        root.addView(pick)
        root.addView(imageView)
        setContentView(ScrollView(this).apply { addView(root) })

        background.execute { loadAndRunSample() }
    }

    private fun loadAndRunSample() {
        try {
            extractor = Dinov2Features(this)
            bitmap = assets.open("sample.jpg").use { BitmapFactory.decodeStream(it) }
            runOnUiThread { status.text = "Ready — extracting features…" }
            runFeatures()
        } catch (e: Throwable) {
            Log.e(TAG, "load failed", e)
            runOnUiThread {
                status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                status.text = "Load failed: ${e.message}"
            }
        }
    }

    private fun pickImage() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
        }
        startActivityForResult(intent, PICK_IMAGE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) {
            return
        }
        val uri: Uri = data?.data ?: return
        contentResolver.openInputStream(uri).use { bitmap = BitmapFactory.decodeStream(it) }
        runFeatures()
    }

    private fun runFeatures() {
        val model = extractor ?: return
        val bm = bitmap ?: return
        runOnUiThread { status.text = "Running on GPU…" }
        background.execute {
            try {
                val start = System.nanoTime()
                val featureMap = model.featureMap(bm)
                val ms = (System.nanoTime() - start) / 1_000_000
                val side = sideBySide(bm, featureMap)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "✓ ${ms}ms · DINOv2 ViT-S/14 features + PCA · CompiledModel GPU"
                    imageView.setImageBitmap(side)
                }
            } catch (e: Throwable) {
                Log.e(TAG, "features failed", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "Failed: ${e.message}"
                }
            }
        }
    }

    /** Draws the source image and the upscaled feature map side by side. */
    private fun sideBySide(source: Bitmap, featureMap: Bitmap): Bitmap {
        val side = 512
        val out = Bitmap.createBitmap(side * 2, side, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawBitmap(
            Bitmap.createScaledBitmap(source, side, side, true), 0f, 0f, null)
        canvas.drawBitmap(
            Bitmap.createScaledBitmap(featureMap, side, side, false), side.toFloat(), 0f, null)
        return out
    }

    override fun onDestroy() {
        super.onDestroy()
        background.shutdown()
        extractor?.close()
    }
}
