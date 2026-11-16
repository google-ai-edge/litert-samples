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

package com.google.ai.edge.examples.ocr_verification

import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.webkit.WebView
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.json.JSONArray
import org.json.JSONObject

/**
 * Photo -> text lines (ML Kit) -> for each line, the OCR text drawn as a picture (WebView) and
 * scored by MiniCPM-OCR-Jev on the GPU (LiteRT CompiledModel): a score from 0 to 1 that the text
 * matches the photo. Lines at or above the threshold are accepted, the rest go to review. Every
 * line's text can be edited and scored again.
 */
class MainActivity : ComponentActivity() {
  private val main = Handler(Looper.getMainLooper())
  private val worker = Executors.newSingleThreadExecutor()
  @Volatile private var verifier: Verifier? = null
  private lateinit var renderer: WebRenderer
  private val regions = TextRegions()
  private var photo: Photo? = null
  private val rows = ArrayList<Row>()
  private var threshold = 0.5
  private var busy = false
  private var runId = 0
  private val prefs by lazy { getSharedPreferences("ocr_check", Context.MODE_PRIVATE) }

  private lateinit var status: TextView
  private lateinit var summary: TextView
  private lateinit var image: ImageView
  private lateinit var overlay: BoxOverlay
  private lateinit var list: LinearLayout
  private lateinit var thresholdLabel: TextView
  private lateinit var buttons: List<Button>

  /** One text line of the photo and its state. */
  private inner class Row(val index: Int, val line: TextLine) {
    var category = RenderPage.Category.TEXT
    /** The text and category the shown score is for (null before the first score). */
    var scoredText: String? = null
    var scoredCategory: RenderPage.Category? = null
    var prob: Double? = null
    var error: String? = null
    var ms: Double? = null
    var buildMs = 0.0
    var running = false
    lateinit var text: EditText
    lateinit var score: TextView
    lateinit var mark: TextView
    lateinit var again: Button
    lateinit var bar: ScoreBar

    /** The text or the category changed after the shown score. */
    val stale: Boolean
      get() = scoredText != null && (text.text.toString() != scoredText || category != scoredCategory)
  }

  private val takePicture =
    registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
      // the camera can outlive this activity: the photo is read from its fixed path, not a field
      if (ok) openFile(cameraFile()) else setStatus("no photo taken")
    }

  private val pickPhoto =
    registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
      if (uri != null) {
        runCatching { Photo.fromUri(contentResolver, uri, "photo") }
          .onSuccess { startRun(it) }
          .onFailure { showProblem("Cannot read that picture: ${it.message}") }
      }
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    // before the first WebView of the process: lets the renderer draw the whole page
    WebView.enableSlowWholeDocumentDraw()
    enableEdgeToEdge(
      statusBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
      navigationBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
    )
    super.onCreate(savedInstanceState)
    val root = FrameLayout(this).apply { setBackgroundColor(Color.WHITE) }
    setContentView(root)
    renderer = WebRenderer(this, root)
    root.addView(buildScreen(), FrameLayout.LayoutParams(-1, -1))
    if (AutoRun.wants(intent)) AutoRun.wakeScreen(this)
    loadModel()
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    if (AutoRun.wants(intent)) {
      AutoRun.wakeScreen(this)
      if (verifier != null && !busy) AutoRun.start(this, intent)
    }
  }

  override fun onDestroy() {
    runId++
    worker.execute {
      verifier?.close()
      verifier = null
    }
    worker.shutdown()
    regions.close()
    renderer.destroy()
    super.onDestroy()
  }

  // ------------------------------------------------------------------ screen

  private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

  private fun buildScreen(): View {
    // system-bar and keyboard insets go on this outer frame, not on the scrolling column
    val outer = FrameLayout(this).apply { setBackgroundColor(Color.WHITE) }
    ViewCompat.setOnApplyWindowInsetsListener(outer) { v, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
      v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
      insets
    }
    val column =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(12f).toInt(), dp(8f).toInt(), dp(12f).toInt(), 0)
        isFocusableInTouchMode = true
        defaultFocusHighlightEnabled = false
      }
    outer.addView(column, FrameLayout.LayoutParams(-1, -1))
    column.addView(
      TextView(this).apply {
        text = getString(R.string.app_name)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 20f)
        setTypeface(typeface, Typeface.BOLD)
        setTextColor(Color.BLACK)
      }
    )
    status =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(Color.DKGRAY)
      }
    column.addView(status)
    setStatus("starting")
    val bar = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    fun button(label: String, onClick: () -> Unit) =
      Button(this).apply {
        text = label
        isAllCaps = false
        isEnabled = false
        setOnClickListener { onClick() }
        bar.addView(this, LinearLayout.LayoutParams(0, -2, 1f))
      }
    buttons =
      listOf(
        button("Camera") { takePicture.launch(cameraUri()) },
        button("Photos") {
          pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        },
        button("Example") { chooseExample() },
      )
    column.addView(bar)
    val photoArea = FrameLayout(this)
    image = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
    overlay = BoxOverlay(this)
    photoArea.addView(image, FrameLayout.LayoutParams(-1, -1))
    photoArea.addView(overlay, FrameLayout.LayoutParams(-1, -1))
    column.addView(photoArea, LinearLayout.LayoutParams(-1, (resources.displayMetrics.heightPixels * 0.24f).toInt()))
    summary =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        setTextColor(Color.BLACK)
        setPadding(0, dp(6f).toInt(), 0, 0)
      }
    column.addView(summary)
    val thresholdRow =
      LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
      }
    thresholdLabel =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        setTextColor(Color.DKGRAY)
      }
    val seek =
      SeekBar(this).apply {
        max = 100
        progress = (threshold * 100).toInt()
        setOnSeekBarChangeListener(
          object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
              threshold = p / 100.0
              refreshAll()
            }

            override fun onStartTrackingTouch(s: SeekBar?) {}

            override fun onStopTrackingTouch(s: SeekBar?) {}
          }
        )
      }
    thresholdRow.addView(thresholdLabel)
    thresholdRow.addView(seek, LinearLayout.LayoutParams(0, -2, 1f))
    column.addView(thresholdRow)
    list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    val scroll =
      ScrollView(this).apply {
        addView(list)
        defaultFocusHighlightEnabled = false
      }
    column.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
    resetToStart()
    return outer
  }

  private fun setStatus(s: String) {
    status.text = "${DeviceInfo.displayName()} · GPU · LiteRT ${BuildConfig.LITERT_VERSION} · $s"
  }

  private fun showProblem(s: String) {
    summary.text = s
    summary.setTextColor(REVIEW)
  }

  private fun setBusy(b: Boolean) {
    busy = b
    buttons.forEach { it.isEnabled = !b && verifier != null }
    rows.forEach { refreshRow(it) }
  }

  /** The screen after opening the app: no photo, no result. */
  fun resetToStart() {
    runId++
    photo = null
    rows.clear()
    list.removeAllViews()
    image.setImageDrawable(null)
    overlay.set(null, emptyList())
    summary.setTextColor(Color.BLACK)
    summary.text = "Take or choose a photo of printed text: every line the text recognizer reads gets a score from 0 to 1 for its text."
    refreshAll()
    if (verifier != null) setStatus("ready")
  }

  // ------------------------------------------------------------------ model

  private fun loadModel() {
    setBusy(true)
    val dir = ModelFiles.dir(filesDir)
    val verified = HashMap(prefs.all.mapValues { it.value.toString() })
    setStatus("checking the model files")
    worker.execute {
      val t0 = System.nanoTime()
      try {
        val missing = ModelFiles.check(dir, hash = false, verified = verified)
        if (missing.isNotEmpty()) throw IllegalStateException(missing.joinToString("\n") { it.message } + "\n\n" + INSTALL_HINT)
        main.post { setStatus("checking the files' SHA-256 (once after each install)") }
        val problems = ModelFiles.check(dir, hash = true, verified = verified)
        if (problems.isNotEmpty()) throw IllegalStateException(problems.joinToString("\n") { it.message } + "\n\n" + INSTALL_HINT)
        prefs.edit().apply { verified.forEach { (k, v) -> putString(k, v) } }.apply()
        // nothing is loaded yet: check the largest set the app keeps built at once, plus a reserve
        val towers = BucketGraphs.discover(dir, BucketGraphs.Kind.TOWER)
        val decides = BucketGraphs.discover(dir, BucketGraphs.Kind.DECIDE)
        val need =
          decides.values.maxOf { it.length() } +
            towers.values.map { it.length() }.sortedDescending().take(2).sum() +
            RESERVE_BYTES
        val available = availableBytes(need)
        if (available in 0 until need) {
          throw IllegalStateException(
            "%.1f GB of memory available, %.1f GB needed. Close other apps and open this app again. Nothing was loaded."
              .format(available / GB, need / GB)
          )
        }
        main.post { setStatus("building the graphs on the GPU") }
        val v = Verifier(dir)
        val built = v.warm(towers.keys.take(2), decides.keys.first())
        verifier = v
        Log.i(
          TAG,
          "LOADED in %.0f ms, builds %s, mem_available_kb %d, self_kb %s, towers %s, decides %s".format(
            (System.nanoTime() - t0) / 1e6, built, DeviceInfo.memAvailableKb(), DeviceInfo.selfStatusKb(),
            towers.keys, decides.keys,
          ),
        )
        main.post {
          setStatus("ready")
          setBusy(false)
          if (AutoRun.wants(intent)) AutoRun.start(this, intent)
        }
      } catch (e: Throwable) {
        Log.e(TAG, "LOAD failed", e)
        main.post {
          setStatus("not ready")
          showProblem(e.message ?: e.toString())
          busy = false
        }
      }
    }
  }

  /** MemAvailable lags a release elsewhere: read it again for up to 3 s before refusing. */
  private fun availableBytes(need: Long): Long {
    var available = DeviceInfo.memAvailableKb() * 1024
    val until = System.nanoTime() + 3_000_000_000L
    while (available in 0 until need && System.nanoTime() < until) {
      Thread.sleep(300)
      available = DeviceInfo.memAvailableKb() * 1024
    }
    return available
  }

  // ------------------------------------------------------------------ input

  private fun cameraFile() = File(cacheDir, "camera/photo.jpg").also { it.parentFile?.mkdirs() }

  private fun cameraUri(): Uri = FileProvider.getUriForFile(this, "$packageName.files", cameraFile())

  private fun openFile(file: File, dump: Boolean = false, then: (() -> Unit)? = null) {
    runCatching { Photo.fromFile(file) }
      .onSuccess { startRun(it, dump, then) }
      .onFailure {
        showProblem("Cannot read the photo: ${it.message}")
        then?.invoke()
      }
  }

  /** A picture a computer copied into files/inputs/ (the adb runs). */
  fun openInput(name: String, dump: Boolean = false, then: (() -> Unit)? = null) =
    openFile(File(filesDir, "inputs/$name"), dump, then)

  private fun examples(): List<Pair<String, String>> {
    val doc = JSONObject(assets.open("examples/examples.json").bufferedReader().readText())
    val arr = doc.getJSONArray("examples")
    return List(arr.length()) { arr.getJSONObject(it).let { e -> e.getString("file") to e.getString("title") } }
  }

  private fun chooseExample() {
    val ex = examples()
    AlertDialog.Builder(this)
      .setTitle("Example pages")
      .setItems(ex.map { it.second }.toTypedArray()) { _, which -> openExample(ex[which].first) }
      .show()
  }

  fun openExample(file: String, dump: Boolean = false, then: (() -> Unit)? = null) {
    runCatching { Photo.fromAsset(this, "examples/$file") }
      .onSuccess { startRun(it, dump, then) }
      .onFailure {
        showProblem("Cannot read $file: ${it.message}")
        then?.invoke()
      }
  }

  // ------------------------------------------------------------------ run

  /** The numbers of the current run (written to files/runs/ when it ends). */
  private var record = JSONObject()
  private var dumpDir: File? = null

  private fun startRun(p: Photo, dump: Boolean = false, then: (() -> Unit)? = null) {
    if (busy || verifier == null) {
      then?.invoke()
      return
    }
    resetToStart()
    photo = p
    val id = runId
    dumpDir = if (dump) File(filesDir, "check/run_${p.name.substringBeforeLast('.')}").apply { deleteRecursively() } else null
    image.setImageBitmap(p.bitmap)
    overlay.set(p, emptyList())
    summary.text = "Reading the text…"
    setStatus("reading the text")
    setBusy(true)
    record =
      JSONObject()
        .put("photo", p.name)
        .put("size", "${p.width}x${p.height}")
        .put("device", DeviceInfo.displayName())
        .put("litert", BuildConfig.LITERT_VERSION)
        .put("threshold", threshold)
        .put("started_ms", System.currentTimeMillis())
        .put("rows", JSONArray())
    val t = System.nanoTime()
    regions.lines(p.bitmap) { result ->
      if (id != runId) return@lines
      result
        .onSuccess { lines ->
          val ocrMs = (System.nanoTime() - t) / 1e6
          record.put("ocr_ms", ocrMs).put("lines", lines.size)
          Log.i(TAG, "OCR ${lines.size} lines in %.0f ms (${p.width}x${p.height}, ${p.name})".format(ocrMs))
          lines.forEachIndexed { i, line -> Row(i, line).also { addRowView(it); rows.add(it) } }
          overlay.set(p, rows)
          if (rows.isEmpty()) {
            summary.text = "No text found in this photo."
            finishRun(then)
          } else {
            setStatus("scoring")
            scoreRows(0, id, then)
          }
        }
        .onFailure {
          showProblem("Text recognition failed: ${it.message}")
          finishRun(then)
        }
    }
  }

  private fun scoreRows(i: Int, id: Int, then: (() -> Unit)?) {
    if (id != runId) return
    if (i >= rows.size) {
      finishRun(then)
      return
    }
    scoreRow(rows[i]) { scoreRows(i + 1, id, then) }
  }

  private fun finishRun(then: (() -> Unit)?) {
    setBusy(false)
    setStatus("ready")
    refreshAll()
    record.put("finished_ms", System.currentTimeMillis())
    val dir = File(filesDir, "runs").apply { mkdirs() }
    File(dir, "run_${record.getLong("started_ms")}.json").writeText(record.toString(1))
    Log.i(
      TAG,
      "RUN done ${record.optString("photo")} ${rows.size} lines, accepted ${rows.count { (it.prob ?: -1.0) >= threshold }} at %.2f"
        .format(threshold),
    )
    then?.invoke()
  }

  /** Crop, render, score one line: the time shown for the line covers all three, on this phone. */
  private fun scoreRow(row: Row, then: () -> Unit) {
    val p = photo
    val v = verifier
    if (p == null || v == null) {
      then()
      return
    }
    val id = runId
    val text = row.text.text.toString()
    val category = row.category
    row.running = true
    refreshRow(row)
    val t0 = System.nanoTime()
    val (region, cropBox) = p.crop(row.line.box)
    val cropMs = (System.nanoTime() - t0) / 1e6
    renderer.render(text, category, cropBox.width()) { rendered ->
      if (id != runId) return@render
      val r =
        rendered.getOrElse {
          row.running = false
          row.error = "render failed: ${it.message}"
          refreshRow(row)
          then()
          return@render
        }
      val lineDir = dumpDir?.let { File(it, "line%02d".format(row.index + 1)) }
      worker.execute {
        val result = runCatching {
          lineDir?.let {
            AutoRun.savePng(region, File(it, "region.png"))
            AutoRun.savePng(r.rgb, File(it, "render.png"))
            File(it, "text.txt").writeText(text, Charsets.UTF_8)
            File(it, "category.txt").writeText(category.name.lowercase())
            AutoRun.writeArray(File(it, "region_input_u8.bin"), region.bytes)
            AutoRun.writeArray(File(it, "render_input_u8.bin"), r.rgb.bytes)
          }
          val dump: ((String, Any) -> Unit)? = lineDir?.let { d -> { name, arr -> AutoRun.writeArray(File(d, "$name.bin"), arr) } }
          v.score(region, r.rgb, text, dump).also { s ->
            lineDir?.let { File(it, "score.json").writeText(AutoRun.scoreJson(s, region, r.rgb).toString(1)) }
          }
        }
        main.post {
          if (id != runId) return@post
          row.running = false
          row.scoredText = text
          row.scoredCategory = category
          result
            .onSuccess { s -> onScored(row, s, text, category, cropBox, cropMs, r) }
            .onFailure { e ->
              row.prob = null
              row.ms = null
              row.error = (e as? Refused)?.let { "Not scored: ${it.message}" } ?: "Not scored: ${e.message}"
              Log.w(TAG, "LINE ${row.index + 1} refused ${(e as? Refused)?.status ?: "error"}: ${e.message}", e)
            }
          refreshRow(row)
          refreshAll()
          then()
        }
      }
    }
  }

  private fun onScored(
    row: Row,
    s: Score,
    text: String,
    category: RenderPage.Category,
    box: android.graphics.Rect,
    cropMs: Double,
    r: WebRenderer.Rendered,
  ) {
    row.prob = s.prob1
    row.error = null
    row.ms = cropMs + r.ms + s.totalMs
    row.buildMs = s.buildMs
    Log.i(
      TAG,
      "LINE %d/%d prob %.5f conf %.3f ms %.1f (crop %.1f, render %.1f, %s) build %.0f T %d n %s tower %s decide L%d category %s text=%s ocr=%s"
        .format(
          row.index + 1, rows.size, s.prob1, row.line.confidence, row.ms, cropMs, r.ms,
          s.ms.entries.joinToString(", ") { e -> "%s %.1f".format(e.key, e.value) },
          s.buildMs, s.tokens, s.patches.toList(), s.towerSizes.toList(), s.decideSize, category.name,
          JSONObject.quote(text), JSONObject.quote(row.line.text),
        ),
    )
    record.getJSONArray("rows").put(
      JSONObject()
        .put("index", row.index + 1)
        .put("text", text)
        .put("ocr_text", row.line.text)
        .put("ocr_confidence", row.line.confidence.toDouble())
        .put("category", category.name)
        .put("box", JSONArray(listOf(box.left, box.top, box.right, box.bottom)))
        .put("render_size", "${r.rgb.width}x${r.rgb.height}")
        .put("katex_errors", JSONArray(r.errors))
        .put("prob_1", s.prob1)
        .put("logits", JSONArray(s.logits.map { it.toDouble() }))
        .put("T", s.tokens)
        .put("n", JSONArray(s.patches.toList()))
        .put("tower_sizes", JSONArray(s.towerSizes.toList()))
        .put("decide_size", s.decideSize)
        .put("ms_line", row.ms)
        .put("ms_crop", cropMs)
        .put("ms_render", r.ms)
        .put("ms_score", JSONObject(s.ms as Map<*, *>))
        .put("build_ms", s.buildMs)
        .put("mem_available_kb", DeviceInfo.memAvailableKb())
        .put("thermal", DeviceInfo.thermalStatus(this))
        .put("cpuset", DeviceInfo.cpuset())
    )
  }

  // ------------------------------------------------------------------ rows

  private fun addRowView(row: Row) {
    // two compact lines per row, so that a page's rows and their marks fit on one screen
    val box =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(0, dp(2f).toInt(), 0, dp(4f).toInt())
      }
    val top =
      LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
      }
    top.addView(
      TextView(this).apply {
        text = "${row.index + 1}"
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(Color.WHITE)
        gravity = Gravity.CENTER
        background =
          GradientDrawable().apply {
            cornerRadius = dp(10f)
            setColor(Color.DKGRAY)
          }
      },
      LinearLayout.LayoutParams(dp(20f).toInt(), dp(20f).toInt()),
    )
    row.text =
      EditText(this).apply {
        setText(row.line.text)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
        setTextColor(Color.BLACK)
        isSingleLine = false
        setPadding(dp(4f).toInt(), dp(4f).toInt(), dp(4f).toInt(), dp(6f).toInt())
        addTextChangedListener(
          object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}

            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}

            override fun afterTextChanged(s: Editable?) = refreshRow(row)
          }
        )
      }
    top.addView(row.text, LinearLayout.LayoutParams(0, -2, 1f).apply { leftMargin = dp(6f).toInt() })
    box.addView(top)
    val bottom =
      LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(26f).toInt(), 0, 0, 0)
      }
    // the text recognizer's own confidence for its line, next to the model's score
    bottom.addView(
      TextView(this).apply {
        text = "OCR conf %.2f".format(row.line.confidence)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(Color.GRAY)
      }
    )
    // the kind of the line (how its text is drawn): a tap opens the three kinds
    val kind =
      TextView(this).apply {
        text = "${row.category.label} ▾"
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(Color.DKGRAY)
        setPadding(dp(8f).toInt(), dp(4f).toInt(), dp(4f).toInt(), dp(4f).toInt())
      }
    kind.setOnClickListener {
      val menu = android.widget.PopupMenu(this, kind)
      RenderPage.Category.values().forEachIndexed { i, c -> menu.menu.add(0, i, i, c.label) }
      menu.setOnMenuItemClickListener { item ->
        val c = RenderPage.Category.values()[item.itemId]
        if (c != row.category) {
          row.category = c
          kind.text = "${c.label} ▾"
          refreshRow(row)
        }
        true
      }
      menu.show()
    }
    bottom.addView(kind)
    row.bar = ScoreBar(this)
    bottom.addView(row.bar, LinearLayout.LayoutParams(dp(40f).toInt(), dp(8f).toInt()).apply { leftMargin = dp(2f).toInt() })
    row.score =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
        setTypeface(typeface, Typeface.BOLD)
        setTextColor(Color.BLACK)
        setPadding(dp(6f).toInt(), 0, dp(2f).toInt(), 0)
      }
    bottom.addView(row.score)
    row.mark =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        setPadding(dp(4f).toInt(), 0, dp(2f).toInt(), 0)
      }
    bottom.addView(row.mark, LinearLayout.LayoutParams(0, -2, 1f))
    row.again =
      Button(this).apply {
        text = "Score again"
        isAllCaps = false
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        minHeight = 0
        minimumHeight = 0
        setPadding(dp(8f).toInt(), dp(2f).toInt(), dp(8f).toInt(), dp(2f).toInt())
        visibility = View.GONE
        setOnClickListener {
          if (!busy) {
            setBusy(true)
            setStatus("scoring")
            scoreRow(row) {
              setBusy(false)
              setStatus("ready")
            }
          }
        }
      }
    bottom.addView(row.again, LinearLayout.LayoutParams(-2, dp(30f).toInt()))
    box.addView(bottom)
    list.addView(box)
    refreshRow(row)
  }

  private fun refreshRow(row: Row) {
    val p = row.prob
    val accepted = p != null && p >= threshold
    val stale = row.stale
    row.score.text =
      when {
        row.running -> "…"
        p == null -> ""
        else -> "%.2f".format(p)
      }
    row.mark.text =
      when {
        row.running -> "scoring"
        stale -> "edited"
        row.error != null -> row.error
        p == null -> ""
        else -> {
          val ms = row.ms?.let { " · %.0f ms".format(it) } ?: ""
          val build = if (row.buildMs > 0) " (+%.1f s GPU build, once)".format(row.buildMs / 1000) else ""
          (if (accepted) "Accept" else "Review") + ms + build
        }
      }
    val color =
      when {
        row.running || stale || p == null -> Color.GRAY
        accepted -> ACCEPT
        else -> REVIEW
      }
    row.mark.setTextColor(if (row.error != null && !stale && !row.running) REVIEW else color)
    row.score.setTextColor(if (stale) Color.GRAY else Color.BLACK)
    row.bar.set(if (row.running) null else p, color)
    row.again.visibility = if (stale && !row.running) View.VISIBLE else View.GONE
    row.again.isEnabled = !busy
    overlay.invalidate()
  }

  private fun refreshAll() {
    thresholdLabel.text = "Accept at ≥ %.2f ".format(threshold)
    rows.forEach { refreshRow(it) }
    val scored = rows.filter { it.prob != null && !it.running }
    if (scored.isEmpty() && rows.isNotEmpty() && !busy) {
      summary.setTextColor(Color.BLACK)
      summary.text =
        if (rows.size == 1) "1 line, not scored (see the line)" else "${rows.size} lines, none scored (see each line)"
    }
    if (scored.isNotEmpty() && !busy) {
      val accepted = scored.count { it.prob!! >= threshold }
      val times = scored.mapNotNull { it.ms }.sorted()
      val median =
        when {
          times.isEmpty() -> null
          times.size % 2 == 1 -> times[times.size / 2]
          else -> (times[times.size / 2 - 1] + times[times.size / 2]) / 2
        }
      summary.setTextColor(Color.BLACK)
      summary.text =
        "$accepted accepted, ${scored.size - accepted} to review" +
          (median?.let { " · median %.0f ms a line".format(it) } ?: "")
    }
  }

  /** A small bar of the score, filled from the left. */
  private inner class ScoreBar(context: Context) : View(context) {
    private var value: Double? = null
    private var color = Color.GRAY
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    fun set(v: Double?, c: Int) {
      value = v
      color = c
      invalidate()
    }

    override fun onDraw(canvas: Canvas) {
      val r = height / 2f
      paint.color = Color.rgb(228, 228, 228)
      canvas.drawRoundRect(0f, 0f, width.toFloat(), height.toFloat(), r, r, paint)
      val v = value ?: return
      paint.color = color
      canvas.drawRoundRect(0f, 0f, (width * v.toFloat()).coerceAtLeast(height.toFloat()), height.toFloat(), r, r, paint)
    }
  }

  /** Line boxes over the photo, coloured by their mark. */
  private inner class BoxOverlay(context: Context) : View(context) {
    private var p: Photo? = null
    private var shown: List<Row> = emptyList()
    private val stroke =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
      }
    private val label =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = dp(11f)
        color = Color.WHITE
      }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)

    fun set(photo: Photo?, rows: List<Row>) {
      p = photo
      shown = rows
      invalidate()
    }

    override fun onDraw(canvas: Canvas) {
      val photo = p ?: return
      val s = minOf(width.toFloat() / photo.width, height.toFloat() / photo.height)
      val ox = (width - photo.width * s) / 2
      val oy = (height - photo.height * s) / 2
      for (row in shown) {
        val b = row.line.box
        val prob = row.prob
        val color =
          when {
            prob == null || row.stale -> Color.GRAY
            prob >= threshold -> ACCEPT
            else -> REVIEW
          }
        stroke.color = color
        val r = RectF(ox + b.left * s, oy + b.top * s, ox + b.right * s, oy + b.bottom * s)
        canvas.drawRect(r, stroke)
        fill.color = color
        val tag = "${row.index + 1}"
        val w = label.measureText(tag) + dp(6f)
        canvas.drawRect(r.left - w, r.top, r.left, r.top + dp(14f), fill)
        canvas.drawText(tag, r.left - w + dp(3f), r.top + dp(11f), label)
      }
    }
  }

  // ------------------------------------------------------------------ check mode

  fun verifierOrNull(): Verifier? = verifier

  fun rendererOrNull(): WebRenderer = renderer

  fun workerExecutor(): ExecutorService = worker

  companion object {
    const val TAG = "OcrCheck"
    private const val GB = 1024.0 * 1024 * 1024
    /** Memory kept free beyond the graph files while they are built. */
    private const val RESERVE_BYTES = 800L * 1024 * 1024
    private const val INSTALL_HINT =
      "Copy the model files to the phone with install_to_device.sh (see the README), then open the app again."
    val ACCEPT = Color.rgb(0x1B, 0x7F, 0x3B)
    val REVIEW = Color.rgb(0xC2, 0x41, 0x0C)
  }
}
