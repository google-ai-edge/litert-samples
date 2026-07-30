// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Bonsai — minimal on-device text-to-image app around BonsaiPipeline.
// UI is deliberately small: prompt, steps, seed, one button, progress, image.
// Generated PNGs land in getExternalFilesDir()/outputs (adb-pullable, and the
// share button exports via FileProvider). CLI runs:
//   adb shell am start -n com.google.ai.edge.samples.imagegeneration/.MainActivity \
//     --ez autorun true --es prompt "a bonsai" --el seed 7 --ei steps 4

package com.google.ai.edge.samples.imagegeneration

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.util.TypedValue
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var promptEdit: EditText
    private lateinit var seedEdit: EditText
    private lateinit var stepsGroup: RadioGroup
    private lateinit var generateButton: Button
    private lateinit var status: TextView
    private lateinit var progress: ProgressBar
    private lateinit var imageView: ImageView
    private lateinit var caption: TextView
    private lateinit var shareButton: Button
    private lateinit var logView: TextView

    private var pipeline: BonsaiPipeline? = null
    private var tokenizer: QwenTokenizer? = null
    private var meta: JSONObject? = null
    private var lastPng: File? = null
    @Volatile private var running = false

    private val modelsDir: File
        get() = getExternalFilesDir(null)!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        meta = JSONObject(assets.open("pipeline_meta.json").bufferedReader().readText())
        checkAssetsAndMaybeAutorun()
    }

    private fun dp(v: Int) = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics
    ).toInt()

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }

        root.addView(TextView(this).apply {
            text = "Bonsai"
            textSize = 28f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })

        promptEdit = EditText(this).apply {
            hint = "Describe an image…"
            setText("a small bonsai tree in a blue ceramic pot")
            minLines = 2
        }
        root.addView(promptEdit)

        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        row.addView(TextView(this).apply { text = "Steps " })
        stepsGroup = RadioGroup(this).apply {
            orientation = RadioGroup.HORIZONTAL
            for (s in listOf(2, 4, 6, 8)) {
                addView(RadioButton(context).apply {
                    text = s.toString()
                    id = s
                })
            }
            check(4)
        }
        row.addView(stepsGroup)
        root.addView(row)

        val seedRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        seedRow.addView(TextView(this).apply { text = "Seed " })
        seedEdit = EditText(this).apply {
            hint = "random"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        seedRow.addView(seedEdit)
        root.addView(seedRow)

        generateButton = Button(this).apply {
            text = "Generate"
            setOnClickListener { if (running) cancel() else generate() }
        }
        root.addView(generateButton)

        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            visibility = android.view.View.GONE
        }
        root.addView(progress)

        status = TextView(this).apply { textSize = 13f }
        root.addView(status)

        imageView = ImageView(this).apply {
            adjustViewBounds = true
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(imageView)

        caption = TextView(this).apply { textSize = 12f }
        root.addView(caption)

        shareButton = Button(this).apply {
            text = "Share"
            visibility = android.view.View.GONE
            setOnClickListener { share() }
        }
        root.addView(shareButton)

        logView = TextView(this).apply {
            textSize = 11f
            typeface = android.graphics.Typeface.MONOSPACE
        }
        root.addView(logView)

        setContentView(ScrollView(this).apply { addView(root) })
    }

    private fun checkAssetsAndMaybeAutorun() {
        val missing = BonsaiPipeline.missingFiles(modelsDir, meta!!)
        if (missing.isNotEmpty()) {
            status.text = "Model files missing from ${modelsDir.path}:\n" +
                missing.joinToString("\n") +
                "\n\nadb push them from " +
                "huggingface.co/litert-community/Bonsai-Image-ternary-4B"
            generateButton.isEnabled = false
            return
        }
        generateButton.isEnabled = true
        if (intent?.getBooleanExtra("autorun", false) == true && !running) {
            intent.getStringExtra("prompt")?.let { promptEdit.setText(it) }
            if (intent.hasExtra("seed")) {
                seedEdit.setText(intent.getLongExtra("seed", 0).toString())
            }
            val s = intent.getIntExtra("steps", 0)
            if (s in listOf(2, 4, 6, 8)) stepsGroup.check(s)
            generate()
        }
    }

    private fun log(line: String) = runOnUiThread {
        android.util.Log.i("bonsai", line)
        logView.append(line + "\n")
    }

    private fun cancel() {
        pipeline?.cancelled = true
        status.text = "Cancelling after this step…"
    }

    private fun generate() {
        if (running) return
        running = true
        generateButton.text = "Cancel"
        imageView.setImageBitmap(null)
        shareButton.visibility = android.view.View.GONE
        logView.text = ""
        progress.visibility = android.view.View.VISIBLE
        val prompt = promptEdit.text.toString().trim()
        val seed = seedEdit.text.toString().toLongOrNull()
            ?: (0..999_999L).random().also { seedEdit.setText(it.toString()) }
        val steps = stepsGroup.checkedRadioButtonId.takeIf { it > 0 } ?: 4

        thread {
            try {
                val tok = tokenizer ?: QwenTokenizer(
                    assets.open("vocab.json"), assets.open("merges.txt")
                ).also { tokenizer = it }
                val pipe = pipeline ?: BonsaiPipeline(modelsDir, meta!!).also { pipeline = it }
                val result = pipe.generate(
                    tokenizer = tok, prompt = prompt, seed = seed, steps = steps,
                    status = { log(it) },
                    progress = { label, f ->
                        runOnUiThread {
                            status.text = label
                            progress.progress = (f * 100).toInt()
                        }
                    })
                val bmp = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
                val px = IntArray(512 * 512)
                for (p in px.indices) {
                    px[p] = Color.rgb(
                        result.rgb[p * 3].toInt() and 0xFF,
                        result.rgb[p * 3 + 1].toInt() and 0xFF,
                        result.rgb[p * 3 + 2].toInt() and 0xFF
                    )
                }
                bmp.setPixels(px, 0, 512, 0, 0, 512, 512)
                val outDir = File(modelsDir, "outputs").apply { mkdirs() }
                val png = File(outDir, "bonsai_seed${seed}_${System.currentTimeMillis()}.png")
                png.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
                lastPng = png
                log("TOTAL %.1fs -> ${png.name}".format(result.seconds))
                runOnUiThread {
                    imageView.setImageBitmap(bmp)
                    caption.text = "seed $seed · $steps steps · %.0f s".format(result.seconds)
                    shareButton.visibility = android.view.View.VISIBLE
                }
            } catch (e: BonsaiPipeline.Cancelled) {
                log("cancelled")
            } catch (e: Exception) {
                log("FAILED: $e")
            } finally {
                running = false
                runOnUiThread {
                    generateButton.text = "Generate"
                    progress.visibility = android.view.View.GONE
                    if (status.text.startsWith("Cancel")) status.text = ""
                }
            }
        }
    }

    private fun share() {
        val png = lastPng ?: return
        val uri: Uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", png)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "Share image"))
    }
}
