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

    private val tag = "Matcha"
    private val bg = Executors.newSingleThreadExecutor()
    private var g2p: MatchaG2P? = null
    private var tts: MatchaSynthesizer? = null

    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var speak: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(56, 120, 56, 56)
        }
        status = TextView(this).apply { textSize = 15f; text = "Loading Matcha-TTS on GPU…" }
        input = EditText(this).apply {
            setText("Hello, this is Matcha running on the mobile GPU.")
            textSize = 16f
        }
        speak = Button(this).apply {
            text = "▶ Speak"; isEnabled = false
            setOnClickListener { synth(input.text.toString()) }
        }
        root.addView(status); root.addView(input); root.addView(speak)
        setContentView(root)

        bg.execute {
            try {
                val gp = MatchaG2P(this); g2p = gp
                val t = MatchaSynthesizer(this); tts = t
                // warm up GPU + JIT
                t.synthesize(gp.phonemize("warm up"), nSteps = 2)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "On-device Matcha-TTS ✓ (GPU graphs + CPU G2P)\nType text and tap Speak."
                    speak.isEnabled = true
                }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "FAIL: ${e.message}\n\nPush models first:\n  scripts/install_to_device.sh"
                }
            }
        }
    }

    private fun synth(text: String) {
        val gp = g2p ?: return
        val t = tts ?: return
        speak.isEnabled = false
        status.text = "Synthesizing…"
        bg.execute {
            try {
                val ids = gp.phonemize(text)
                val r = t.synthesize(ids)
                val dur = r.audio.size.toDouble() / MatchaSynthesizer.SAMPLE_RATE
                val rtf = r.ms / 1000.0 / dur
                Log.i(tag, "phonemes=${ids.size} frames=${r.frames} steps=${r.steps} ${r.ms}ms rtf=$rtf")
                runOnUiThread {
                    status.text = "Spoke ${"%.2f".format(dur)} s in ${r.ms} ms · RTF ${"%.2f".format(rtf)} " +
                        "(${r.steps} ODE steps, ${ids.size} phonemes)"
                    speak.isEnabled = true
                }
                play(r.audio)
            } catch (e: Throwable) {
                Log.e(tag, "synth failed", e)
                runOnUiThread { status.text = "FAIL: ${e.message}"; speak.isEnabled = true }
            }
        }
    }

    private fun play(audio: FloatArray) {
        try {
            val track = AudioTrack(
                AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
                AudioFormat.Builder()
                    .setSampleRate(MatchaSynthesizer.SAMPLE_RATE)
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
                audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
            )
            track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
            track.play()
            Thread.sleep((audio.size * 1000L / MatchaSynthesizer.SAMPLE_RATE) + 250)
            track.release()
        } catch (e: Throwable) {
            Log.e(tag, "play failed: ${e.message}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        tts?.close(); g2p?.close()
    }
}
