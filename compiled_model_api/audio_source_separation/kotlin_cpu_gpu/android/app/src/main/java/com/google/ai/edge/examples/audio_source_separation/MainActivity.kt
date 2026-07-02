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

package com.google.ai.edge.examples.audio_source_separation

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
import android.widget.ProgressBar
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * TIGER-DnR cinematic sound separation, fully on-device GPU: pick (or record) a clip and split it
 * into Dialogue / Sound effects / Music stems. Three ~1.4 M-param TIGER graphs run per 12 s chunk
 * on the LiteRT CompiledModel GPU; host does only reflect-pad, iSTFT and overlap-add.
 */
class MainActivity : Activity() {

    private val tag = "TIGER"
    private val bg = Executors.newSingleThreadExecutor()
    private var sep: TigerSeparator? = null

    private lateinit var status: TextView
    private lateinit var progress: ProgressBar
    private val stemButtons = HashMap<String, Button>()
    private var mixture: FloatArray? = null
    private var stems: List<FloatArray>? = null
    private var player: AudioTrack? = null
    @Volatile private var recording = false

    private val maxSeconds = 32   // 3 chunks (12.06 s window, 10 s hop)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(36, 90, 36, 36) }
        status = TextView(this).apply { textSize = 15f; text = "Loading…" }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100; visibility = android.view.View.GONE
        }
        val pick = Button(this).apply {
            text = "🎬  Pick audio / video clip"
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                    putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "video/*"))
                }, 1)
            }
        }
        val rec = Button(this).apply {
            text = "🎙  Record 15 s"
            setOnClickListener { toggleRecord(this) }
        }
        root.addView(status); root.addView(pick); root.addView(rec); root.addView(progress)

        val label = TextView(this).apply { textSize = 14f; text = "Stems (first ${maxSeconds}s):"; setPadding(0, 36, 0, 8) }
        root.addView(label)
        val names = mapOf("mixture" to "▶  Mixture", "dialog" to "🗣  Dialogue",
                          "effect" to "💥  Sound effects", "music" to "🎵  Music")
        for ((key, title) in names) {
            val b = Button(this).apply { text = title; isEnabled = false; setOnClickListener { play(key, this) } }
            stemButtons[key] = b
            root.addView(b)
        }
        setContentView(root)

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)

        bg.execute {
            try {
                sep = TigerSeparator(this)
                // fail fast if models are missing
                runOnUiThread { status.text = "Ready — pick a clip (movie scene, game, vlog…) to separate." }
            } catch (e: Throwable) {
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val uri = data?.data ?: return
        if (requestCode != 1 || resultCode != RESULT_OK) return
        bg.execute {
            try {
                runOnUiThread { status.text = "Decoding…"; setStemsEnabled(false) }
                val pcm = AudioDecoder.decode(this, uri, maxSeconds)
                check(pcm.size >= TigerSeparator.SR) { "Clip too short (need ≥1 s of audio)." }
                runSeparation(pcm)
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
        btn.text = "⏹  Stop"
        status.text = "● Recording (up to 15 s)…"
        bg.execute {
            val sr = TigerSeparator.SR
            val min = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
            val recd = AudioRecord(MediaRecorder.AudioSource.MIC, sr, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT, maxOf(min, sr * 2))
            val out = FloatArray(sr * 15)
            var total = 0
            try {
                recd.startRecording()
                val buf = FloatArray(4410)
                while (recording && total < out.size) {
                    val r = recd.read(buf, 0, minOf(buf.size, out.size - total), AudioRecord.READ_BLOCKING)
                    if (r > 0) { System.arraycopy(buf, 0, out, total, r); total += r }
                }
            } finally { recd.stop(); recd.release(); recording = false }
            runOnUiThread { btn.text = "🎙  Record 15 s" }
            if (total >= sr) runSeparation(out.copyOf(total))
            else runOnUiThread { status.text = "Too short — record at least 1 s." }
        }
    }

    private fun runSeparation(pcm: FloatArray) {
        val s = sep ?: return
        mixture = pcm
        val t0 = System.nanoTime()
        val total = TigerSeparator.STEMS.size
        val result = s.separate(pcm) { stem, c, n ->
            val stemIdx = TigerSeparator.STEMS.indexOf(stem)
            val pct = ((stemIdx * n + c - 1) * 100) / (total * n)
            runOnUiThread {
                progress.visibility = android.view.View.VISIBLE
                progress.progress = pct
                status.text = "Separating on GPU… $stem $c/$n"
            }
        }
        val ms = (System.nanoTime() - t0) / 1_000_000
        stems = result
        Log.i(tag, "separated ${pcm.size / TigerSeparator.SR}s in ${ms}ms")
        runOnUiThread {
            progress.visibility = android.view.View.GONE
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "Separated ${pcm.size / TigerSeparator.SR}s in ${ms / 1000.0}s · TIGER-DnR, CompiledModel GPU"
            setStemsEnabled(true)
        }
    }

    private fun setStemsEnabled(on: Boolean) {
        for (b in stemButtons.values) b.isEnabled = on
    }

    private fun play(key: String, btn: Button) {
        stopPlayback()
        val data = when (key) {
            "mixture" -> mixture
            else -> stems?.getOrNull(TigerSeparator.STEMS.indexOf(key))
        } ?: return
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(TigerSeparator.SR).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(data.size * 4)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        track.write(data, 0, data.size, AudioTrack.WRITE_BLOCKING)
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
        sep?.close()
    }
}
