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

package com.google.ai.edge.examples.text_to_speech

import android.app.Activity
import android.graphics.Color
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * Matcha-TTS demo: type text, synthesize it on the GPU, and play it back. All three heavy
 * graphs (text encoder, CFM decoder, HiFi-GAN vocoder) run on the LiteRT CompiledModel GPU;
 * the G2P runs on the CompiledModel CPU. No network, no espeak.
 */
class MainActivity : Activity() {

    private val backgroundExecutor = Executors.newSingleThreadExecutor()
    private var g2p: MatchaG2P? = null
    private var synthesizer: MatchaSynthesizer? = null

    private lateinit var statusText: TextView
    private lateinit var inputText: EditText
    private lateinit var speakButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(PADDING_PX, PADDING_TOP_PX, PADDING_PX, PADDING_PX)
        }
        statusText = TextView(this).apply {
            textSize = 15f
            text = "Loading Matcha-TTS on GPU…"
        }
        inputText = EditText(this).apply {
            setText("Hello, this is Matcha running on the mobile GPU.")
            textSize = 16f
        }
        speakButton = Button(this).apply {
            text = "▶ Speak"
            isEnabled = false
            setOnClickListener { synthesizeAndPlay(inputText.text.toString()) }
        }
        root.addView(statusText)
        root.addView(inputText)
        root.addView(speakButton)
        setContentView(root)

        backgroundExecutor.execute { loadModels() }
    }

    /** Loads the G2P and synthesizer models, then warms up the GPU with a short utterance. */
    private fun loadModels() {
        try {
            val loadedG2p = MatchaG2P(this)
            g2p = loadedG2p
            val loadedSynthesizer = MatchaSynthesizer(this)
            synthesizer = loadedSynthesizer
            loadedSynthesizer.synthesize(loadedG2p.phonemize("warm up"), nSteps = WARM_UP_STEPS)
            runOnUiThread {
                statusText.setBackgroundColor(STATUS_OK_COLOR)
                statusText.text =
                    "On-device Matcha-TTS ✓ (GPU graphs + CPU G2P)\nType text and tap Speak."
                speakButton.isEnabled = true
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Model load failed", e)
            runOnUiThread {
                statusText.setBackgroundColor(STATUS_ERROR_COLOR)
                statusText.text =
                    "FAIL: ${e.message}\n\nPush models first:\n  scripts/install_to_device.sh"
            }
        }
    }

    /** Phonemizes [text], synthesizes it off the UI thread, and plays the result. */
    private fun synthesizeAndPlay(text: String) {
        val loadedG2p = g2p ?: return
        val loadedSynthesizer = synthesizer ?: return
        speakButton.isEnabled = false
        statusText.text = "Synthesizing…"
        backgroundExecutor.execute {
            try {
                val phonemeIds = loadedG2p.phonemize(text)
                val result = loadedSynthesizer.synthesize(phonemeIds)
                val durationSec = result.audio.size.toDouble() / MatchaSynthesizer.SAMPLE_RATE
                val realTimeFactor = result.ms / 1000.0 / durationSec
                Log.i(
                    TAG,
                    "phonemes=${phonemeIds.size} frames=${result.frames} steps=${result.steps} " +
                        "${result.ms}ms rtf=$realTimeFactor",
                )
                runOnUiThread {
                    statusText.text =
                        "Spoke ${"%.2f".format(durationSec)} s in ${result.ms} ms · " +
                            "RTF ${"%.2f".format(realTimeFactor)} " +
                            "(${result.steps} ODE steps, ${phonemeIds.size} phonemes)"
                    speakButton.isEnabled = true
                }
                play(result.audio)
            } catch (e: Throwable) {
                Log.e(TAG, "Synthesis failed", e)
                runOnUiThread {
                    statusText.text = "FAIL: ${e.message}"
                    speakButton.isEnabled = true
                }
            }
        }
    }

    /** Plays mono float PCM at [MatchaSynthesizer.SAMPLE_RATE] and blocks until done. */
    private fun play(audio: FloatArray) {
        try {
            val track = AudioTrack(
                AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
                AudioFormat.Builder()
                    .setSampleRate(MatchaSynthesizer.SAMPLE_RATE)
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
                audio.size * BYTES_PER_FLOAT,
                AudioTrack.MODE_STATIC,
                AudioManager.AUDIO_SESSION_ID_GENERATE,
            )
            track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
            track.play()
            Thread.sleep((audio.size * 1000L / MatchaSynthesizer.SAMPLE_RATE) + PLAYBACK_TAIL_MS)
            track.release()
        } catch (e: Throwable) {
            Log.e(TAG, "Playback failed: ${e.message}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        synthesizer?.close()
        g2p?.close()
    }

    companion object {
        private const val TAG = "Matcha"
        private const val PADDING_PX = 56
        private const val PADDING_TOP_PX = 120
        private const val WARM_UP_STEPS = 2
        private const val BYTES_PER_FLOAT = 4
        /** Extra wait after the nominal audio duration so the tail is not cut off. */
        private const val PLAYBACK_TAIL_MS = 250L
        private val STATUS_OK_COLOR = Color.rgb(0xC8, 0xE6, 0xC9)
        private val STATUS_ERROR_COLOR = Color.rgb(0xFF, 0xCD, 0xD2)
    }
}
