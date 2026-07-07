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

package com.google.ai.edge.examples.music_transcription

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * Basic Pitch on-device: play an instrument (or sing / pick a clip) and see the transcribed
 * notes on a piano roll. Fully-GPU model incl. the conv-CQT front-end (~4 ms per 2 s window).
 */
class MainActivity : Activity() {

    private val tag = "BasicPitch"
    private val bg = Executors.newSingleThreadExecutor()
    private var transcriber: Transcriber? = null

    private lateinit var status: TextView
    private lateinit var roll: PianoRollView
    private lateinit var summary: TextView
    @Volatile private var recording = false
    private val maxSeconds = 30

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
        val rec = Button(this).apply {
            text = "🎹  Record & transcribe"
            setOnClickListener { toggleRecord(this) }
        }
        val pick = Button(this).apply {
            text = "📁  Pick audio / video clip"
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                    putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "video/*"))
                }, 1)
            }
        }
        roll = PianoRollView(this)
        summary = TextView(this).apply {
            textSize = 14f
            setPadding(0, 16, 0, 0)
        }
        root.addView(status)
        root.addView(rec)
        root.addView(pick)
        root.addView(roll)
        root.addView(summary)
        setContentView(ScrollView(this).apply { addView(root) })

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)

        bg.execute {
            try {
                transcriber = Transcriber(this)
                runOnUiThread { status.text = "Ready — play some notes and record, or pick a clip." }
            } catch (e: Throwable) {
                Log.e(tag, "load", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "FAIL: ${e.message}"
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val uri = data?.data ?: return
        if (requestCode != 1 || resultCode != RESULT_OK) return
        bg.execute {
            try {
                runOnUiThread { status.text = "Decoding…" }
                val x = AudioDecoder.decode(this, uri, maxSeconds)
                check(x.size >= Transcriber.SR / 2) { "Clip too short." }
                run(x)
            } catch (e: Throwable) {
                Log.e(tag, "pick", e)
                runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun toggleRecord(btn: Button) {
        if (recording) {
            recording = false
            return
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            status.text = "Microphone permission needed."
            return
        }
        recording = true
        btn.text = "⏹  Stop & transcribe"
        status.text = "● Recording (up to ${maxSeconds}s)…"
        bg.execute {
            val sr = Transcriber.SR
            val min = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
            // Buffer size is in BYTES and must be a multiple of the 4-byte float frame;
            // reserve at least one second (sr frames * 4 bytes).
            val recd = AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, sr, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT, maxOf(min, sr * Float.SIZE_BYTES))
            val out = FloatArray(sr * maxSeconds)
            var total = 0
            try {
                recd.startRecording()
                val buf = FloatArray(2205)
                while (recording && total < out.size) {
                    val r = recd.read(buf, 0, minOf(buf.size, out.size - total), AudioRecord.READ_BLOCKING)
                    if (r > 0) {
                        System.arraycopy(buf, 0, out, total, r)
                        total += r
                    }
                }
            } finally { recd.stop()
            recd.release()
            recording = false }
            runOnUiThread { btn.text = "🎹  Record & transcribe" }
            if (total >= sr / 2) run(out.copyOf(total))
            else runOnUiThread { status.text = "Too short." }
        }
    }

    private fun run(x: FloatArray) {
        val tr = transcriber ?: return
        val t0 = System.nanoTime()
        val (note, onset) = tr.posteriorgrams(x) { w, n ->
            runOnUiThread { status.text = "Transcribing on GPU… window $w/$n" }
        }
        val events = tr.decode(note, onset)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "transcribed ${x.size / Transcriber.SR}s in ${ms}ms: ${events.size} notes")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "✓ ${events.size} notes from ${x.size / Transcriber.SR}s in ${ms / 1000.0}s · Basic Pitch, CompiledModel GPU"
            roll.set(events, x.size.toDouble() / Transcriber.SR)
            summary.text = events.take(24).joinToString("  ") {
                "${roll.noteName(it.midi)}@%.1fs".format(it.startSec)
            } + if (events.size > 24) "  …" else ""
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        recording = false
        bg.shutdown()
        transcriber?.close()
    }
}
