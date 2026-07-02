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

package com.google.ai.edge.examples.speech_enhancement

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * CMGAN noise suppression, fully on-device GPU: record (or pick) a noisy clip and A/B the
 * denoised result. One 1.83 M-param conformer graph per 2 s chunk (~20 ms/chunk on a Pixel 8a).
 */
class MainActivity : Activity() {

    private val tag = "CMGAN"
    private val bg = Executors.newSingleThreadExecutor()
    private var ns: NoiseSuppressor? = null

    private lateinit var status: TextView
    private lateinit var playNoisy: Button
    private lateinit var playClean: Button
    private var noisy: FloatArray? = null
    private var clean: FloatArray? = null
    private var player: AudioTrack? = null
    @Volatile private var recording = false

    private val maxSeconds = 30

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(36, 90, 36, 36) }
        status = TextView(this).apply { textSize = 15f; text = "Loading…" }
        val rec = Button(this).apply { text = "🎙  Record noisy audio"; setOnClickListener { toggleRecord(this) } }
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
        playNoisy = Button(this).apply { text = "▶  Noisy (original)"; isEnabled = false; setOnClickListener { play(noisy) } }
        playClean = Button(this).apply { text = "✨  Enhanced"; isEnabled = false; setOnClickListener { play(clean) } }
        root.addView(status); root.addView(rec); root.addView(pick); root.addView(playNoisy); root.addView(playClean)
        setContentView(root)

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)

        bg.execute {
            try {
                ns = NoiseSuppressor(this)
                runOnUiThread { status.text = "Ready — record in a noisy place, or pick a clip." }
            } catch (e: Throwable) {
                Log.e(tag, "load", e)
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
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
                check(x.size >= NoiseSuppressor.SR) { "Clip too short (need ≥1 s)." }
                run(x)
            } catch (e: Throwable) {
                Log.e(tag, "pick", e)
                runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun toggleRecord(btn: Button) {
        if (recording) { recording = false; return }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            status.text = "Microphone permission needed."; return
        }
        recording = true
        btn.text = "⏹  Stop & enhance"
        status.text = "● Recording (up to ${maxSeconds}s)…"
        bg.execute {
            val sr = NoiseSuppressor.SR
            val min = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
            val recd = AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, sr, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT, maxOf(min, sr * 2))
            val out = FloatArray(sr * maxSeconds)
            var total = 0
            try {
                recd.startRecording()
                val buf = FloatArray(1600)
                while (recording && total < out.size) {
                    val r = recd.read(buf, 0, minOf(buf.size, out.size - total), AudioRecord.READ_BLOCKING)
                    if (r > 0) { System.arraycopy(buf, 0, out, total, r); total += r }
                }
            } finally { recd.stop(); recd.release(); recording = false }
            runOnUiThread { btn.text = "🎙  Record noisy audio" }
            if (total >= sr) run(out.copyOf(total))
            else runOnUiThread { status.text = "Too short — record at least 1 s." }
        }
    }

    private fun run(x: FloatArray) {
        val n = ns ?: return
        noisy = x
        val t0 = System.nanoTime()
        val y = n.enhance(x) { c, total ->
            runOnUiThread { status.text = "Enhancing on GPU… chunk $c/$total" }
        }
        val ms = (System.nanoTime() - t0) / 1_000_000
        clean = y
        Log.i(tag, "enhanced ${x.size / NoiseSuppressor.SR}s in ${ms}ms")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "✓ ${x.size / NoiseSuppressor.SR}s enhanced in ${ms / 1000.0}s · CMGAN, CompiledModel GPU"
            playNoisy.isEnabled = true
            playClean.isEnabled = true
        }
    }

    private fun play(data: FloatArray?) {
        data ?: return
        stopPlayback()
        var peak = 1e-6f
        for (v in data) if (kotlin.math.abs(v) > peak) peak = kotlin.math.abs(v)
        val scaled = FloatArray(data.size) { data[it] / peak * 0.9f }
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(NoiseSuppressor.SR).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(scaled.size * 4)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        track.write(scaled, 0, scaled.size, AudioTrack.WRITE_BLOCKING)
        track.play()
        player = track
    }

    private fun stopPlayback() {
        player?.let { runCatching { it.stop(); it.release() } }
        player = null
    }

    override fun onDestroy() {
        super.onDestroy()
        recording = false
        stopPlayback()
        bg.shutdown()
        ns?.close()
    }
}
