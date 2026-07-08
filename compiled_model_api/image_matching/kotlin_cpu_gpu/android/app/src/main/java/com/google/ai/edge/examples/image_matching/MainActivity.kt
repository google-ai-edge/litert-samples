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

package com.google.ai.edge.examples.image_matching

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * XFeat two-image matching, fully on-device GPU: pick two photos of the same scene (different
 * angles) and see the matched keypoints. ~0.4 ms/image on the GPU; decode + matching on the host.
 */
class MainActivity : Activity() {

    private val tag = "XFeat"
    private val bg = Executors.newSingleThreadExecutor()
    private var matcher: XFeatMatcher? = null

    private lateinit var status: TextView
    private lateinit var view: MatchView
    private var bmA: Bitmap? = null
    private var bmB: Bitmap? = null

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
        val pickA = Button(this).apply {
            text = "1️⃣  Pick first image"
            setOnClickListener { pick(1) }
        }
        val pickB = Button(this).apply {
            text = "2️⃣  Pick second image"
            setOnClickListener { pick(2) }
        }
        view = MatchView(this)
        root.addView(status)
        root.addView(pickA)
        root.addView(pickB)
        root.addView(view)
        setContentView(ScrollView(this).apply { addView(root) })

        bg.execute {
            try {
                matcher = XFeatMatcher(this)
                runOnUiThread { status.text = "Ready — pick two photos of the same scene." }
            } catch (e: Throwable) {
                Log.e(tag, "load", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "FAIL: ${e.message}"
                }
            }
        }
    }

    private fun pick(which: Int) {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
        }, which)
    }

    private fun load(uri: Uri): Bitmap {
        contentResolver.openInputStream(uri).use { s ->
            return BitmapFactory.decodeStream(s)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val uri = data?.data ?: return
        if (resultCode != RESULT_OK) return
        try {
            val bm = load(uri)
            if (requestCode == 1) bmA = bm else bmB = bm
            status.text = "Image $requestCode set (${bm.width}x${bm.height})."
            val a = bmA
            val b = bmB
            if (a != null && b != null) bg.execute { run(a, b) }
        } catch (e: Throwable) {
            Log.e(tag, "pick", e)
            status.text = "Failed: ${e.message}"
        }
    }

    private fun run(a: Bitmap, b: Bitmap) {
        val m = matcher ?: return
        runOnUiThread { status.text = "Matching on GPU…" }
        val t0 = System.nanoTime()
        val fa = m.extract(m.preprocess(a))
        val fb = m.extract(m.preprocess(b))
        val matches = m.match(fa, fb)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "kpts ${fa.xs.size}/${fb.xs.size}, ${matches.size} matches in ${ms}ms")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "✓ ${matches.size} matches (${fa.xs.size}/${fb.xs.size} kpts) " +
                "in ${ms}ms · XFeat, CompiledModel GPU"
            view.set(Bitmap.createScaledBitmap(a, XFeatMatcher.W, XFeatMatcher.H, true),
                     Bitmap.createScaledBitmap(b, XFeatMatcher.W, XFeatMatcher.H, true), matches)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        matcher?.close()
    }
}
