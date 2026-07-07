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

package com.google.ai.edge.examples.text_to_speech_lm

import android.app.Activity
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * Qwen3-TTS demo: type text, synthesize it fully on device with the LiteRT
 * Compiled Model host loop (talker LM + MTP + codec decoder on CPU), and play
 * it back. Model files are pushed with install_to_device.sh; see the README.
 */
class MainActivity : Activity() {

    private val tag = "Qwen3Tts"
    private val bg = Executors.newSingleThreadExecutor()
    private var engine: Qwen3TtsEngine? = null

    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var language: Spinner
    private lateinit var speak: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(56, 120, 56, 56)
        }
        status = TextView(this).apply {
            textSize = 15f
            text = "Loading Qwen3-TTS (three graphs, ~1.2 GB)…"
        }
        input = EditText(this).apply {
            setText("Hello! This is Qwen3 text to speech running on device.")
            textSize = 16f
        }
        language = Spinner(this).apply {
            val langs = listOf("english") +
                (Qwen3TtsEngine.LANGUAGE_IDS.keys - "english").sorted() + listOf("auto")
            adapter = ArrayAdapter(
                this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
                langs)
        }
        speak = Button(this).apply {
            text = "Speak"
            isEnabled = false
            setOnClickListener { synthesize() }
        }
        root.addView(status)
        root.addView(input)
        root.addView(language)
        root.addView(speak)
        setContentView(root)

        bg.execute {
            try {
                val t0 = System.currentTimeMillis()
                val e = Qwen3TtsEngine(filesDir)
                engine = e
                val tokenizerOk = tokenizerSelfTest(e)
                val ms = System.currentTimeMillis() - t0
                runOnUiThread {
                    status.text = "Ready (loaded in ${ms / 1000.0}s" +
                        (if (tokenizerOk) "" else "; TOKENIZER SELF-TEST FAILED, see logcat") +
                        "). CPU-only; expect a few seconds per second of audio."
                    speak.isEnabled = true
                }
            } catch (t: Throwable) {
                Log.e(tag, "load failed", t)
                runOnUiThread { status.text = "Load failed: ${t.message}" }
            }
        }
    }

    /** Compares the Kotlin BPE against reference ids baked at conversion time. */
    private fun tokenizerSelfTest(e: Qwen3TtsEngine): Boolean {
        return try {
            val json = assets.open("tokenizer_test_vectors.json")
                .bufferedReader().readText()
            val cases = org.json.JSONArray(json)
            var ok = true
            for (i in 0 until cases.length()) {
                val case = cases.getJSONObject(i)
                val want = case.getJSONArray("ids")
                val got = e.encodeText(case.getString("text"))
                val same = got.size == want.length() &&
                    got.indices.all { got[it] == want.getInt(it) }
                if (!same) {
                    ok = false
                    Log.e(tag, "tokenizer mismatch for: ${case.getString("text")}\n" +
                        "want=${(0 until want.length()).map { want.getInt(it) }}\n" +
                        "got =${got.toList()}")
                }
            }
            Log.i(tag, "tokenizer self-test: ${if (ok) "PASS" else "FAIL"}")
            ok
        } catch (t: Throwable) {
            Log.e(tag, "tokenizer self-test error", t)
            false
        }
    }

    private fun synthesize() {
        val text = input.text.toString()
        val lang = language.selectedItem as String
        speak.isEnabled = false
        status.text = "Generating…"
        bg.execute {
            try {
                val e = engine!!
                val result = e.synthesize(
                    text, language = lang,
                    progress = object : Qwen3TtsEngine.Progress {
                        override fun onFrame(frame: Int) {
                            if (frame % 5 == 0) {
                                runOnUiThread {
                                    status.text =
                                        "Generating… $frame frames (${"%.1f".format(frame / 12.5)}s)"
                                }
                            }
                        }
                    })
                val seconds = result.audio.size / Qwen3TtsEngine.SAMPLE_RATE.toFloat()
                val wall = result.prefillMs + result.talkerMs + result.mtpMs +
                    result.codecMs
                Log.i(tag, "frames=${result.frames} prefill=${result.prefillMs}ms " +
                    "talker=${result.talkerMs}ms mtp=${result.mtpMs}ms " +
                    "codec=${result.codecMs}ms")
                runOnUiThread {
                    status.text = "Spoke %.2fs in %.1fs (RTF %.1f) — talker %dms, mtp %dms, codec %dms"
                        .format(seconds, wall / 1000.0, wall / 1000.0 / seconds,
                            result.talkerMs, result.mtpMs, result.codecMs)
                    speak.isEnabled = true
                }
                play(result.audio)
            } catch (t: Throwable) {
                Log.e(tag, "synthesis failed", t)
                runOnUiThread {
                    status.text = "Failed: ${t.message}"
                    speak.isEnabled = true
                }
            }
        }
    }

    private fun play(audio: FloatArray) {
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build())
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(Qwen3TtsEngine.SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build())
            .setBufferSizeInBytes(audio.size * 4)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
        track.play()
    }

    override fun onDestroy() {
        super.onDestroy()
        engine?.close()
    }
}
