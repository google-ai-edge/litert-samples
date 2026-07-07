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

package com.google.ai.edge.examples.speaker_diarization

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
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * On-device speaker diarization ("who spoke when"): record a conversation (or pick a clip) and
 * get a per-speaker timeline. PyanNet segmentation runs on CPU (onnxruntime, BiLSTM); WeSpeaker
 * ResNet34 embeddings run fully on the LiteRT CompiledModel GPU; clustering is host-side.
 */
class MainActivity : Activity() {

    private val tag = "Diarization"
    private val bg = Executors.newSingleThreadExecutor()
    private var diarizer: Diarizer? = null

    private lateinit var status: TextView
    private lateinit var summary: TextView
    private lateinit var timeline: TimelineView
    private lateinit var speakerButtons: LinearLayout
    private var pcm: FloatArray? = null
    private var result: Diarizer.Result? = null
    private var player: AudioTrack? = null
    @Volatile private var recording = false

    private val maxSeconds = 120

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
            text = "🎙  Record conversation"
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
        timeline = TimelineView(this)
        summary = TextView(this).apply {
            textSize = 15f
            setPadding(0, 24, 0, 0)
        }
        speakerButtons = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(status)
        root.addView(rec)
        root.addView(pick)
        root.addView(timeline)
        root.addView(summary)
        root.addView(speakerButtons)
        setContentView(ScrollView(this).apply { addView(root) })

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)

        bg.execute {
            try {
                diarizer = Diarizer(this)
                runOnUiThread { status.text = "Ready — record a conversation or pick a clip." }
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
                check(x.size >= Diarizer.SR * 2) { "Clip too short (need ≥2 s)." }
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
        btn.text = "⏹  Stop & analyze"
        status.text = "● Recording (up to ${maxSeconds}s)…"
        bg.execute {
            val sr = Diarizer.SR
            val min = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
            val recd = AudioRecord(MediaRecorder.AudioSource.MIC, sr, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT, maxOf(min, sr * 2))
            val out = FloatArray(sr * maxSeconds)
            var total = 0
            try {
                recd.startRecording()
                val buf = FloatArray(1600)
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
            runOnUiThread { btn.text = "🎙  Record conversation" }
            if (total >= sr * 2) run(out.copyOf(total))
            else runOnUiThread { status.text = "Too short — record at least 2 s." }
        }
    }

    private fun run(x: FloatArray) {
        val d = diarizer ?: return
        pcm = x
        val t0 = System.nanoTime()
        val res = d.diarize(x) { msg -> runOnUiThread { status.text = msg } }
        val ms = (System.nanoTime() - t0) / 1_000_000
        result = res
        Log.i(tag, "diarized ${x.size / Diarizer.SR}s in ${ms}ms: ${res.numSpeakers} speakers, " +
                res.segments.joinToString { "%.1f-%.1f:S%d".format(it.start, it.end, it.speaker) })
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "✓ ${res.numSpeakers} speakers · ${x.size / Diarizer.SR}s in ${ms / 1000.0}s " +
                    "(seg CPU + embeddings GPU)"
            timeline.set(res.segments, x.size.toDouble() / Diarizer.SR)
            summary.text = res.perSpeaker.toSortedMap().entries
                .joinToString("\n") { (spk, dur) -> "SPK ${spk + 1}: %.1f s".format(dur) }
            speakerButtons.removeAllViews()
            for (spk in res.perSpeaker.keys.sorted()) {
                speakerButtons.addView(Button(this).apply {
                    text = "▶  Play SPK ${spk + 1}"
                    setTextColor(TimelineView.COLORS[spk % TimelineView.COLORS.size])
                    setOnClickListener { playSpeaker(spk) }
                })
            }
        }
    }

    private fun playSpeaker(spk: Int) {
        val x = pcm ?: return
        val res = result ?: return
        stopPlayback()
        val segs = res.segments.filter { it.speaker == spk }
        if (segs.isEmpty()) return
        val n = segs.sumOf { ((it.end - it.start) * Diarizer.SR).toInt() }
        val data = FloatArray(n)
        var pos = 0
        for (s in segs) {
            val a = (s.start * Diarizer.SR).toInt().coerceIn(0, x.size)
            val b = (s.end * Diarizer.SR).toInt().coerceIn(0, x.size)
            for (p in a until b) { if (pos < n) data[pos++] = x[p] }
        }
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(Diarizer.SR).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
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
        diarizer?.close()
    }
}
