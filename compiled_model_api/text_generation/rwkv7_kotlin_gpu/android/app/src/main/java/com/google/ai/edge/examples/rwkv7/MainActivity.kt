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

package com.google.ai.edge.examples.rwkv7

import android.graphics.Typeface
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "Rwkv7"

/**
 * Minimal completion/chat UI over [Rwkv7Generator]: type a prompt, watch tokens
 * stream in as they come off the GPU. Greedy decoding.
 *
 * The 282 MB step graph and the 100 MB embedding table are NOT bundled — run
 * `install_to_device.sh` after installing the APK (the first launch
 * shows instructions until the files are in place).
 */
class MainActivity : ComponentActivity() {

    private companion object {
        const val MODEL_FILE = "rwkv7_step_fp16.tflite"
        const val EMB_FILE = "rwkv7_emb_fp16.bin"
        const val VOCAB_ASSET = "rwkv_vocab_v20230424.txt"
        const val MAX_NEW_TOKENS = 200
        const val DEFAULT_PROMPT = "The Eiffel tower is in the city of"
        const val CHAT_TEMPLATE = "User: %s\n\nAssistant:"
        /** In chat mode a blank line ends the assistant turn. */
        const val CHAT_STOP = "\n\n"
    }

    private lateinit var statusText: TextView
    private lateinit var promptEdit: EditText
    private lateinit var chatModeBox: CheckBox
    private lateinit var generateButton: Button
    private lateinit var outputText: TextView
    private lateinit var outputScroll: ScrollView

    private var generator: Rwkv7Generator? = null
    private var tokenizer: RwkvTokenizer? = null
    private val executor = Executors.newSingleThreadExecutor()
    private val stopRequested = AtomicBoolean(false)
    @Volatile private var generating = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        executor.execute { loadModel() }
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF1A1A1A.toInt())
            setPadding(24, 48, 24, 24)
        }

        statusText = TextView(this).apply {
            textSize = 14f
            setTextColor(0xFFCCCCCC.toInt())
            setPadding(0, 8, 0, 8)
            text = "Loading RWKV-7 0.1B (GPU)..."
        }
        root.addView(statusText)

        promptEdit = EditText(this).apply {
            setBackgroundColor(0xFF2A2A2A.toInt())
            setTextColor(0xFFEEEEEE.toInt())
            setHintTextColor(0xFF888888.toInt())
            setPadding(24, 16, 24, 16)
            hint = "Prompt"
            setText(DEFAULT_PROMPT)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            maxLines = 4
            textSize = 16f
        }
        root.addView(promptEdit)

        chatModeBox = CheckBox(this).apply {
            text = "Chat template (User:/Assistant:)"
            setTextColor(0xFFCCCCCC.toInt())
        }
        root.addView(chatModeBox)

        generateButton = Button(this).apply {
            text = "Generate"
            isEnabled = false
            setOnClickListener { onGenerateClicked() }
        }
        root.addView(generateButton)

        outputText = TextView(this).apply {
            textSize = 16f
            setTextColor(0xFFEEEEEE.toInt())
            typeface = Typeface.MONOSPACE
            setTextIsSelectable(true)
            setPadding(8, 16, 8, 16)
        }
        outputScroll = ScrollView(this).apply { addView(outputText) }
        root.addView(
            outputScroll,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f),
        )

        setContentView(root)
    }

    private fun loadModel() {
        try {
            val vocabLines = assets.open(VOCAB_ASSET).bufferedReader().readLines()
            tokenizer = RwkvTokenizer(vocabLines)
            val modelFile = File(filesDir, MODEL_FILE)
            val embFile = File(filesDir, EMB_FILE)
            if (!modelFile.exists() || !embFile.exists()) {
                runOnUiThread {
                    statusText.text =
                        "Model files missing — run install_to_device.sh, then relaunch."
                }
                return
            }
            generator = Rwkv7Generator(modelFile.absolutePath, embFile.absolutePath)
            runOnUiThread {
                statusText.text = "RWKV-7 World 0.1B ready — full forward on GPU"
                generateButton.isEnabled = true
            }
        } catch (e: Exception) {
            Log.e(TAG, "model load failed", e)
            runOnUiThread { statusText.text = "Load failed: ${e.message}" }
        }
    }

    private fun onGenerateClicked() {
        if (generating) {
            stopRequested.set(true)
            return
        }
        val prompt = promptEdit.text.toString()
        if (prompt.isBlank()) {
            return
        }
        val chatMode = chatModeBox.isChecked
        val fullPrompt = if (chatMode) CHAT_TEMPLATE.format(prompt.trim()) else prompt
        generating = true
        stopRequested.set(false)
        generateButton.text = "Stop"
        outputText.text = fullPrompt
        executor.execute { runGeneration(fullPrompt, chatMode) }
    }

    private fun runGeneration(fullPrompt: String, chatMode: Boolean) {
        val gen = generator ?: return
        val tok = tokenizer ?: return
        val promptIds = tok.encode(fullPrompt)
        val bytes = ByteArrayOutputStream()
        var stepMsSum = 0f
        var count = 0
        val startNs = System.nanoTime()
        gen.generate(promptIds, MAX_NEW_TOKENS) { tokenId, stats ->
            bytes.write(tok.tokenBytes(tokenId))
            val text = bytes.toString(Charsets.UTF_8.name())
            stepMsSum += stats.stepMs
            count++
            runOnUiThread {
                outputText.text = fullPrompt + text
                outputScroll.fullScroll(ScrollView.FOCUS_DOWN)
            }
            val stop = stopRequested.get() || (chatMode && text.endsWith(CHAT_STOP))
            !stop
        }
        val totalS = (System.nanoTime() - startNs) / 1e9f
        val tokPerS = if (totalS > 0f) count / totalS else 0f
        runOnUiThread {
            statusText.text =
                "%d tokens, %.1f tok/s (%.1f ms/token) — prefill %d ids".format(
                    count, tokPerS, if (count > 0) stepMsSum / count else 0f, promptIds.size)
            generateButton.text = "Generate"
        }
        generating = false
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        generator?.close()
    }
}
