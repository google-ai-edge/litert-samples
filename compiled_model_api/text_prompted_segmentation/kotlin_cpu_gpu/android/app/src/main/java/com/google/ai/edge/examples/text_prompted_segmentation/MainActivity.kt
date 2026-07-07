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

package com.google.ai.edge.examples.text_prompted_segmentation

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * CLIPSeg text-prompted segmentation, on-device GPU: pick an image, type what you want to
 * segment ("a dog", "the sky"), and see the mask overlay. Both CLIP encoders + the decoder run
 * on the LiteRT CompiledModel GPU.
 */
class MainActivity : Activity() {

    private val tag = "CLIPSeg"
    private val bg = Executors.newSingleThreadExecutor()
    private var seg: ClipSeg? = null

    private lateinit var status: TextView
    private lateinit var prompt: EditText
    private lateinit var imageView: ImageView
    private var bitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(36, 90, 36, 36)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading…"
        }
        val pick = Button(this).apply {
            text = "🖼  Pick image"
            setOnClickListener { pickImage() }
        }
        prompt = EditText(this).apply {
            hint = "What to segment, e.g. \"a cat\""
            inputType = InputType.TYPE_CLASS_TEXT
            setText("a cat")
        }
        val go = Button(this).apply {
            text = "✂  Segment"
            setOnClickListener { runSegment() }
        }
        imageView = ImageView(this).apply { adjustViewBounds = true }
        root.addView(status)
        root.addView(pick)
        root.addView(prompt)
        root.addView(go)
        root.addView(imageView)
        setContentView(root)

        bg.execute {
            try {
                seg = ClipSeg(this)
                runOnUiThread { status.text = "Ready — pick an image and type a prompt." }
            } catch (e: Throwable) {
                Log.e(tag, "load", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "FAIL: ${e.message}"
                }
            }
        }
    }

    private fun pickImage() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
        }, 1)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val uri = data?.data ?: return
        if (resultCode != RESULT_OK) return
        try {
            bitmap = load(uri)
            imageView.setImageBitmap(bitmap)
            status.text = "Image set (${bitmap!!.width}x${bitmap!!.height}). Type a prompt and Segment."
        } catch (e: Throwable) { status.text = "Failed: ${e.message}" }
    }

    private fun load(uri: Uri): Bitmap {
        contentResolver.openInputStream(uri).use { return BitmapFactory.decodeStream(it) }
    }

    private fun runSegment() {
        val s = seg ?: return
        val bm = bitmap ?: run {
            status.text = "Pick an image first."
            return
        }
        val text = prompt.text.toString().ifBlank { "object" }
        runOnUiThread { status.text = "Segmenting \"$text\" on GPU…" }
        bg.execute {
            try {
                val t0 = System.nanoTime()
                val mask = s.segment(bm, text)
                val ms = (System.nanoTime() - t0) / 1_000_000
                val overlay = overlay(bm, mask)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "✓ \"$text\" in ${ms}ms · CLIPSeg, CompiledModel GPU"
                    imageView.setImageBitmap(overlay)
                }
            } catch (e: Throwable) {
                Log.e(tag, "segment", e)
                runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    /** Blend a red mask over the image (mask is 352x352, resized to the display bitmap). */
    private fun overlay(bm: Bitmap, mask: FloatArray): Bitmap {
        val w = bm.width
        val h = bm.height
        val out = bm.copy(Bitmap.Config.ARGB_8888, true)
        val px = IntArray(w * h)
        out.getPixels(px, 0, w, 0, 0, w, h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val mx = (x * ClipSeg.SIZE / w).coerceIn(0, ClipSeg.SIZE - 1)
                val my = (y * ClipSeg.SIZE / h).coerceIn(0, ClipSeg.SIZE - 1)
                val m = mask[my * ClipSeg.SIZE + mx]
                if (m > 0.1f) {
                    val i = y * w + x
                    val p = px[i]
                    val a = (m * 0.6f).coerceIn(0f, 0.6f)
                    val r = (((p shr 16) and 0xFF) * (1 - a) + 255 * a).toInt()
                    val g = (((p shr 8) and 0xFF) * (1 - a)).toInt()
                    val b = ((p and 0xFF) * (1 - a)).toInt()
                    px[i] = Color.rgb(r, g, b)
                }
            }
        }
        out.setPixels(px, 0, w, 0, 0, w, h)
        return out
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        seg?.close()
    }
}
