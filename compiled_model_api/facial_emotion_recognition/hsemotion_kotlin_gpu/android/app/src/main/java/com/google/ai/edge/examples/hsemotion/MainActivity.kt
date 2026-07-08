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

package com.google.ai.edge.examples.hsemotion

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

private const val TAG = "HSEmotion"
private const val PICK_IMAGE = 1

/**
 * Facial emotion recognition demo. Pick a face photo (or use the bundled sample)
 * and see the predicted emotion distribution. The HSEmotion EfficientNet-B0
 * backbone runs fully on the LiteRT CompiledModel GPU. A face crop and softmax
 * are the only host-side work.
 */
class MainActivity : Activity() {

    private val background = Executors.newSingleThreadExecutor()
    private var classifier: EmotionClassifier? = null
    private var bitmap: Bitmap? = null

    private lateinit var status: TextView
    private lateinit var imageView: ImageView
    private lateinit var resultView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 100, 40, 40)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading HSEmotion…"
        }
        val pick = Button(this).apply {
            text = "🙂  Pick a face photo"
            setOnClickListener { pickImage() }
        }
        imageView = ImageView(this).apply { adjustViewBounds = true }
        resultView = TextView(this).apply {
            textSize = 18f
            setPadding(0, 24, 0, 0)
        }
        root.addView(status)
        root.addView(pick)
        root.addView(imageView)
        root.addView(resultView)
        setContentView(ScrollView(this).apply { addView(root) })

        background.execute { loadAndClassifySample() }
    }

    private fun loadAndClassifySample() {
        try {
            classifier = EmotionClassifier(this)
            bitmap = assets.open("sample.jpg").use { BitmapFactory.decodeStream(it) }
            runOnUiThread {
                imageView.setImageBitmap(bitmap)
                status.text = "Ready — classifying sample…"
            }
            runClassify()
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
        imageView.setImageBitmap(bitmap)
        runClassify()
    }

    private fun runClassify() {
        val model = classifier ?: return
        val bm = bitmap ?: return
        runOnUiThread { status.text = "Classifying on GPU…" }
        background.execute {
            try {
                val start = System.nanoTime()
                val predictions = model.classify(bm)
                val ms = (System.nanoTime() - start) / 1_000_000
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "✓ ${ms}ms · HSEmotion EfficientNet-B0, CompiledModel GPU"
                    resultView.text = predictions.joinToString("\n") {
                        "%5.1f%%  %s".format(it.probability * 100f, it.label)
                    }
                }
            } catch (e: Throwable) {
                Log.e(TAG, "classify failed", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "Failed: ${e.message}"
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        background.shutdown()
        classifier?.close()
    }
}
