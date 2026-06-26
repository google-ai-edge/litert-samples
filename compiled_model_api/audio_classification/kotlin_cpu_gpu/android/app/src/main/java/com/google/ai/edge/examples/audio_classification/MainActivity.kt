/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.audio_classification

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/**
 * wav2vec2 keyword-spotting demo: classifies a bundled 1-second clip on launch (fully on the GPU),
 * then a "Record" button captures 1 s from the mic and classifies the spoken keyword.
 */
class MainActivity : Activity() {

    private val tag = "W2VKWS"
    private val bg = Executors.newSingleThreadExecutor()
    private var kws: Wav2Vec2Kws? = null
    private lateinit var status: TextView
    private lateinit var record: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(56, 120, 56, 56)
        }
        status = TextView(this).apply { textSize = 15f; text = "Loading wav2vec2 KWS on GPU…" }
        record = Button(this).apply {
            text = "🎤 Record 1 s & classify"; isEnabled = false
            setOnClickListener { ensureMicThenRecord() }
        }
        root.addView(status); root.addView(record)
        setContentView(root)

        bg.execute {
            try {
                val k = Wav2Vec2Kws(this); kws = k
                val clip = readFloats(assets.open("test_audio.bin").readBytes())
                k.classify(clip)                       // warm up GPU
                val r = k.classify(clip)               // measured
                Log.i(tag, "bundled clip -> ${r.label} (${r.ms} ms)")
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "On-device GPU keyword spotting ✓\n\n" +
                        "bundled clip → \"${r.label}\"  (${r.ms} ms)\n\n" +
                        "Tap Record and say one of:\n" + Wav2Vec2Kws.LABELS.take(10).joinToString(" · ")
                    record.isEnabled = true
                }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "FAIL: ${e.message}"
                }
            }
        }
    }

    private fun ensureMicThenRecord() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
            return
        }
        recordAndClassify()
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) {
        super.onRequestPermissionsResult(rc, p, g)
        if (rc == 1 && g.isNotEmpty() && g[0] == PackageManager.PERMISSION_GRANTED) recordAndClassify()
    }

    private fun recordAndClassify() {
        record.isEnabled = false
        runOnUiThread { status.text = "Listening… speak now" }
        bg.execute {
            try {
                val audio = recordOneSecond()
                val r = kws!!.classify(audio)
                Log.i(tag, "mic -> ${r.label} (${r.ms} ms)")
                runOnUiThread {
                    status.text = "Heard: \"${r.label}\"  (${r.ms} ms)\n\nTap Record to try again."
                    record.isEnabled = true
                }
            } catch (e: Throwable) {
                Log.e(tag, "record failed", e)
                runOnUiThread { status.text = "Record failed: ${e.message}"; record.isEnabled = true }
            }
        }
    }

    /** Capture 1 s of mono 16 kHz PCM from the mic, returned as float32 in [-1,1]. */
    private fun recordOneSecond(): FloatArray {
        val sr = Wav2Vec2Kws.SAMPLES
        val minBuf = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, sr,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, sr * 2),
        )
        val pcm = ShortArray(sr)
        rec.startRecording()
        var off = 0
        while (off < sr) {
            val n = rec.read(pcm, off, sr - off)
            if (n <= 0) break
            off += n
        }
        rec.stop(); rec.release()
        val out = FloatArray(sr)
        var peak = 1f
        for (i in 0 until sr) peak = maxOf(peak, kotlin.math.abs(pcm[i].toFloat()))
        for (i in 0 until sr) out[i] = pcm[i] / peak * 0.5f   // peak-normalize, match training scale
        return out
    }

    private fun readFloats(b: ByteArray): FloatArray {
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.float }
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        kws?.close()
    }
}
