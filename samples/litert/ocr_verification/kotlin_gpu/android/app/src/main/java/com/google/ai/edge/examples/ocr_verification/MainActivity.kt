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
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextWatcher
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.webkit.WebView
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Switch
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
 * matches the photo. Lines at or above the threshold are accepted, the rest go to review. With the
 * check off, every line is accepted as the recognizer read it. The lines are a table: the line's
 * picture cut from the photo, the text the recognizer read, and the model's score. Every line's
 * text can be edited there (with keys for symbols a phone keyboard hides) and scored again.
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
  /** Score each line with the model; off, every line the recognizer reads is accepted as it is. */
  private var check = true
  private var busy = false
  private var runId = 0
  private val prefs by lazy { getSharedPreferences("ocr_check", Context.MODE_PRIVATE) }

  private lateinit var status: TextView
  /** The count under the photo, in large type: lines accepted and lines to review. */
  private lateinit var tally: TextView
  private lateinit var summary: TextView
  private lateinit var image: ImageView
  private lateinit var overlay: BoxOverlay
  private lateinit var photoArea: FrameLayout
  private lateinit var checkSwitch: Switch
  private lateinit var buttonBar: View
  private lateinit var thresholdRow: View
  /** Shows or folds the photo once its lines are in the table. */
  private lateinit var photoToggle: TextView
  private var photoOpen = true
  private lateinit var tableHeader: View
  private lateinit var tableScroll: ScrollView
  private lateinit var list: LinearLayout
  /** The keys for symbols, shown while a line's text is edited. */
  private lateinit var symbols: View
  /** The line whose text has the focus (being edited), and whether the keyboard is up. */
  private var editing: Row? = null
  private var keyboardUp = false
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
    lateinit var pill: ScorePill
    lateinit var again: Button
    lateinit var frame: View
    /** Shown while the line is edited: its kind, the recognizer's confidence and the time. */
    lateinit var details: View
    lateinit var info: TextView
    /** Why the line was not scored. */
    lateinit var note: TextView
    lateinit var view: View

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
    check = prefs.getBoolean(PREF_CHECK, true)
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
      val up = insets.isVisible(WindowInsetsCompat.Type.ime())
      if (up != keyboardUp) {
        keyboardUp = up
        // after this layout pass: the views above the table fold while a line is edited
        v.post { refreshLines() }
      }
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
    buttonBar = bar
    checkSwitch =
      Switch(this).apply {
        text = "Check each line with MiniCPM-OCR-Jev"
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
        setTextColor(Color.BLACK)
        isChecked = check
        setPadding(dp(4f).toInt(), dp(4f).toInt(), dp(4f).toInt(), dp(4f).toInt())
        setOnCheckedChangeListener { _, on -> setCheck(on, scoreNow = true) }
      }
    column.addView(checkSwitch, LinearLayout.LayoutParams(-1, -2))
    photoToggle =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        setTextColor(Color.rgb(0x1E, 0x5A, 0xA8))
        setPadding(0, dp(2f).toInt(), 0, dp(2f).toInt())
        setOnClickListener {
          photoOpen = !photoOpen
          refreshLines()
        }
      }
    column.addView(photoToggle)
    photoArea = FrameLayout(this)
    image = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
    overlay = BoxOverlay(this)
    photoArea.addView(image, FrameLayout.LayoutParams(-1, -1))
    photoArea.addView(overlay, FrameLayout.LayoutParams(-1, -1))
    column.addView(photoArea, LinearLayout.LayoutParams(-1, emptyPhotoHeight()))
    tally =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 24f)
        setTypeface(typeface, Typeface.BOLD)
        setPadding(0, dp(6f).toInt(), 0, 0)
        visibility = View.GONE
      }
    column.addView(tally)
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
    this.thresholdRow = thresholdRow
    // the table of lines: its heading has the shape of a line (the picture above, the text and
    // the score under it)
    tableHeader =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(0, dp(4f).toInt(), 0, dp(2f).toInt())
        // the column names at the size of the line's text and in the text's own color, so they read
        // at a glance which part is the photo, which the recognizer's text and which the model's score
        fun heading(s: String, g: Int) =
          TextView(this@MainActivity).apply {
            text = s
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.BLACK)
            gravity = g
          }
        addView(heading("Image", Gravity.START))
        addView(
          LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(heading("OCR read", Gravity.START), LinearLayout.LayoutParams(0, -2, 1f))
            addView(heading("Jev score", Gravity.END))
          }
        )
      }
    column.addView(tableHeader)
    list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    tableScroll =
      ScrollView(this).apply {
        addView(list)
        defaultFocusHighlightEnabled = false
      }
    column.addView(tableScroll, LinearLayout.LayoutParams(-1, 0, 1f))
    symbols = symbolKeys()
    column.addView(symbols)
    resetToStart()
    return outer
  }

  /**
   * Keys for symbols that a phone keyboard keeps out of reach (fractions, superscripts, the degree
   * sign, Greek letters): a key puts its symbol at the cursor of the line being edited, or in place
   * of the selected text. The keys never take the focus, so the keyboard stays up.
   */
  private fun symbolKeys(): View {
    val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    for (s in SYMBOLS) {
      row.addView(
        Button(this).apply {
          text = s
          isAllCaps = false
          setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
          minWidth = 0
          minimumWidth = 0
          minHeight = 0
          minimumHeight = 0
          setPadding(0, dp(6f).toInt(), 0, dp(6f).toInt())
          isFocusable = false
          isFocusableInTouchMode = false
          contentDescription = if (s == DELETE) "Delete the character before the cursor" else "Insert $s"
          setOnClickListener { insertSymbol(s) }
        },
        LinearLayout.LayoutParams(0, -2, 1f),
      )
    }
    return row.apply { visibility = View.GONE }
  }

  private fun insertSymbol(s: String) {
    val field = editing?.text ?: return
    val a = minOf(field.selectionStart, field.selectionEnd).coerceAtLeast(0)
    val b = maxOf(field.selectionStart, field.selectionEnd).coerceAtLeast(0)
    when {
      s != DELETE -> field.text.replace(a, b, s)
      b > a -> field.text.delete(a, b)
      a > 0 -> field.text.delete(a - 1, a)
    }
  }

  private fun hideKeyboard(field: View) {
    val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager
    imm.hideSoftInputFromWindow(field.windowToken, 0)
    field.clearFocus()
  }

  /** The photo area's height before a photo is opened. */
  private fun emptyPhotoHeight() = (resources.displayMetrics.heightPixels * 0.24f).toInt()

  /** The photo area takes the photo's shape, up to about half of the screen. */
  private fun fitPhotoArea(p: Photo?) {
    val max = (resources.displayMetrics.heightPixels * 0.5f).toInt()
    val width = if (photoArea.width > 0) photoArea.width else resources.displayMetrics.widthPixels - dp(24f).toInt()
    val h = if (p == null) emptyPhotoHeight() else minOf(max, (width.toLong() * p.height / p.width).toInt() + dp(4f).toInt())
    if (photoArea.layoutParams.height != h) {
      photoArea.layoutParams = photoArea.layoutParams.apply { height = h }
    }
  }

  /**
   * What the screen shows around the table: the photo while its lines are read (then folded, a tap
   * opens it again), the table once there are lines; while a line is edited with the keyboard up,
   * the buttons, the switch, the photo and the threshold fold so that the line stays in view.
   */
  private fun refreshLines() {
    val edit = editing != null && keyboardUp
    val table = rows.isNotEmpty()
    fun show(v: View, on: Boolean) {
      val want = if (on) View.VISIBLE else View.GONE
      if (v.visibility != want) v.visibility = want
    }
    show(buttonBar, !edit)
    show(checkSwitch, !edit)
    show(thresholdRow, !edit)
    show(photoToggle, table && !edit)
    photoToggle.text = if (photoOpen) "Hide the photo ▴" else "Show the photo ▾"
    show(photoArea, (!table || photoOpen) && !edit)
    show(tableHeader, table)
    show(tableScroll, table)
    show(symbols, editing != null)
    editing?.let { r -> tableScroll.post { tableScroll.smoothScrollTo(0, r.view.top) } }
  }

  /** Scrolls the table to [row] (a tap on its box in the photo). */
  private fun showLine(row: Row) {
    tableScroll.post { tableScroll.smoothScrollTo(0, row.view.top) }
  }

  /**
   * Turns the model's check on or off. Off, every line is accepted as the recognizer read it; on,
   * with [scoreNow], the lines of the open photo that have no score yet are scored.
   */
  fun setCheck(on: Boolean, scoreNow: Boolean) {
    if (busy && on != check) {
      // the switch stays as it is while lines are read or scored (it is not greyed out: it shows
      // whether the check is on)
      setSwitch(check)
      return
    }
    setSwitch(on)
    check = on
    prefs.edit().putBoolean(PREF_CHECK, on).apply()
    val start = on && scoreNow && !busy && verifier != null && rows.any { it.prob == null && it.error == null }
    if (start) {
      // busy before the screen is refreshed, so the count goes straight from "accepted" to checking
      setBusy(true)
      setStatus("scoring")
    }
    refreshAll()
    if (start) scoreRows(0, runId, null)
  }

  /** Sets the switch without calling its listener (which comes back to [setCheck]). */
  private fun setSwitch(on: Boolean) {
    if (checkSwitch.isChecked != on) {
      checkSwitch.setOnCheckedChangeListener(null)
      checkSwitch.isChecked = on
      checkSwitch.setOnCheckedChangeListener { _, v -> setCheck(v, scoreNow = true) }
    }
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
    // the count and the line under it say "checking" while busy
    if (!b) refreshAll()
  }

  /** The screen after opening the app: no photo, no result. */
  fun resetToStart() {
    runId++
    photo = null
    rows.clear()
    list.removeAllViews()
    image.setImageDrawable(null)
    overlay.set(null, emptyList())
    fitPhotoArea(null)
    photoOpen = true
    editing = null
    refreshLines()
    tally.visibility = View.GONE
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
    fitPhotoArea(p)
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
          sizeLinePictures()
          overlay.set(p, rows)
          // the table takes the screen: each line's picture is cut from the photo
          photoOpen = false
          refreshLines()
          if (rows.isEmpty()) {
            summary.text = "No text found in this photo."
            finishRun(then)
          } else if (check) {
            setStatus("scoring")
            scoreRows(0, id, then)
          } else {
            finishRun(then)
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
    if (rows[i].prob != null || rows[i].error != null) {
      scoreRows(i + 1, id, then)
      return
    }
    scoreRow(rows[i]) { scoreRows(i + 1, id, then) }
  }

  private fun finishRun(then: (() -> Unit)?) {
    setBusy(false)
    setStatus("ready")
    refreshAll()
    record.put("finished_ms", System.currentTimeMillis()).put("check", check)
    val dir = File(filesDir, "runs").apply { mkdirs() }
    File(dir, "run_${record.getLong("started_ms")}.json").writeText(record.toString(1))
    if (rows.isNotEmpty() && rows.none { it.prob != null || it.error != null }) {
      Log.i(TAG, "RUN unchecked ${record.optString("photo")} ${rows.size} lines (the check is off)")
    } else {
      Log.i(
        TAG,
        "RUN done ${record.optString("photo")} ${rows.size} lines, accepted ${rows.count { (it.prob ?: -1.0) >= threshold }} at %.2f"
          .format(threshold),
      )
    }
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
    refreshAll()
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

  /** A row's gray label: "Image:" beside the line's picture, "OCR read:" beside its text. */
  private fun rowLabel(s: String) =
    TextView(this).apply {
      text = s
      setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
      setTextColor(LABEL)
      setPadding(0, dp(3f).toInt(), dp(4f).toInt(), dp(3f).toInt())
      maxLines = 1
    }

  /** One width for both labels of a row (the wider one and its padding): the labels make a column. */
  private val labelWidth: Int by lazy {
    val paint = rowLabel(TEXT_LABEL).paint
    kotlin.math.ceil(maxOf(paint.measureText(TEXT_LABEL), paint.measureText(IMAGE_LABEL)) + dp(4f)).toInt() + 1
  }

  private fun addRowView(row: Row) {
    // a line of the table: the line's picture (cut from the photo: the region the model reads)
    // above the text the recognizer read and the model's score, each beside its gray label
    val box =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(0, dp(5f).toInt(), 0, 0)
      }
    val picture =
      LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        background =
          GradientDrawable().apply {
            setColor(Color.WHITE)
            setStroke(dp(1f).toInt(), FRAME)
          }
        val pad = dp(1f).toInt()
        setPadding(pad, pad, pad, pad)
      }
    val image = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_XY }
    photo?.let { p ->
      val r = p.crop(row.line.box).second
      image.setImageBitmap(Bitmap.createBitmap(p.bitmap, r.left, r.top, r.width(), r.height()))
      image.tag = r
    }
    // the line's number, small, in the frame's corner, beside the picture (never over its print)
    picture.addView(
      TextView(this).apply {
        text = "${row.index + 1}"
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
        setTextColor(Color.WHITE)
        setPadding(dp(3f).toInt(), 0, dp(3f).toInt(), 0)
        setBackgroundColor(Color.argb(160, 70, 70, 70))
      },
      LinearLayout.LayoutParams(-2, -2).apply { gravity = Gravity.TOP },
    )
    picture.addView(image, LinearLayout.LayoutParams(-2, -2))
    row.frame = picture
    // the picture's label in the column of the text's label below it: the picture and the text start
    // at the same place, and a line read right does not look like the same line shown twice
    val pictureLine =
      LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
      }
    pictureLine.addView(rowLabel(IMAGE_LABEL), LinearLayout.LayoutParams(labelWidth, -2))
    pictureLine.addView(picture, LinearLayout.LayoutParams(-2, -2))
    box.addView(pictureLine)
    val line =
      LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
      }
    row.text =
      EditText(this).apply {
        setText(row.line.text)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
        setTextColor(Color.BLACK)
        isSingleLine = false
        background = null
        setPadding(0, dp(3f).toInt(), dp(4f).toInt(), dp(3f).toInt())
        addTextChangedListener(
          object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}

            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}

            override fun afterTextChanged(s: Editable?) = refreshRow(row)
          }
        )
        setOnFocusChangeListener { _, has ->
          if (has) editing = row else if (editing === row) editing = null
          refreshRow(row)
          refreshLines()
        }
      }
    // what the row's text is, beside the field and not in it: the label is never part of the text
    // that is edited or scored (top-aligned, so its baseline is the first line's when the text wraps)
    line.addView(rowLabel(TEXT_LABEL), LinearLayout.LayoutParams(labelWidth, -2).apply { gravity = Gravity.TOP })
    line.addView(row.text, LinearLayout.LayoutParams(0, -2, 1f).apply { gravity = Gravity.TOP })
    row.pill = ScorePill(this)
    line.addView(row.pill, LinearLayout.LayoutParams(-2, -2).apply { leftMargin = dp(6f).toInt() })
    row.again =
      Button(this).apply {
        text = "Score again"
        isAllCaps = false
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        minHeight = 0
        minimumHeight = 0
        setPadding(dp(10f).toInt(), dp(2f).toInt(), dp(10f).toInt(), dp(2f).toInt())
        // a flat background without the default button's insets: the button is no taller than the line's text, so the
        // rows below do not move down (under the keyboard) when it takes the score's place
        background =
          RippleDrawable(
            ColorStateList.valueOf(Color.rgb(0xBD, 0xBD, 0xBD)),
            GradientDrawable().apply {
              setColor(Color.rgb(0xE0, 0xE0, 0xE0))
              cornerRadius = dp(4f)
            },
            null,
          )
        stateListAnimator = null
        visibility = View.GONE
        setOnClickListener {
          if (!busy) {
            // the keyboard goes down and the screen comes back before the line is scored
            hideKeyboard(row.text)
            setBusy(true)
            setStatus("scoring")
            scoreRow(row) {
              setBusy(false)
              setStatus("ready")
            }
          }
        }
      }
    line.addView(row.again, LinearLayout.LayoutParams(-2, -2).apply { leftMargin = dp(6f).toInt() })
    box.addView(line)
    // why the line was not scored
    row.note =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(REVIEW)
        visibility = View.GONE
      }
    box.addView(row.note)
    // while the line is edited: its kind (how its text is drawn: a tap opens the three kinds), the
    // text recognizer's own confidence for the line and the line's time
    val details =
      LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        visibility = View.GONE
      }
    val kind =
      TextView(this).apply {
        text = "${row.category.label} ▾"
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(Color.DKGRAY)
        setPadding(0, dp(4f).toInt(), dp(8f).toInt(), dp(4f).toInt())
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
    details.addView(kind)
    row.info =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(Color.GRAY)
      }
    details.addView(row.info)
    row.details = details
    box.addView(details)
    box.addView(
      View(this).apply { setBackgroundColor(RULE) },
      LinearLayout.LayoutParams(-1, dp(1f).toInt()).apply { topMargin = dp(4f).toInt() },
    )
    list.addView(box)
    row.view = box
    refreshRow(row)
  }

  /**
   * One scale for every line's picture, so the print keeps one size down the table: the widest
   * picture fills the table's width, and no picture is taller than 56 dp.
   */
  private fun sizeLinePictures() {
    val rects = rows.mapNotNull { (it.frame as LinearLayout).getChildAt(1).tag as? Rect }
    if (rects.isEmpty()) return
    // the table's width less the label column, the frame and the number beside the picture
    val avail = resources.displayMetrics.widthPixels - dp(26f) - dp(18f) - labelWidth
    val s = minOf(avail / rects.maxOf { it.width() }, dp(56f) / rects.maxOf { it.height() })
    for (row in rows) {
      val image = (row.frame as LinearLayout).getChildAt(1)
      val r = image.tag as? Rect ?: continue
      image.layoutParams = image.layoutParams.apply {
        width = (r.width() * s).toInt().coerceAtLeast(1)
        height = (r.height() * s).toInt().coerceAtLeast(1)
      }
    }
  }

  private fun refreshRow(row: Row) {
    val st = state(row)
    when (st) {
      LineState.SCORING -> row.pill.set(null, "checking…", SCORING)
      LineState.ACCEPT -> row.pill.set(row.prob, "Accept", ACCEPT)
      LineState.REVIEW -> row.pill.set(row.prob, "Review", REVIEW)
      LineState.UNCHECKED -> row.pill.set(null, "Accept", ACCEPT)
      LineState.FAILED -> row.pill.set(null, "not scored", REVIEW)
      LineState.WAITING, LineState.EDITED -> row.pill.set(null, null, Color.GRAY)
    }
    row.pill.visibility = if (st == LineState.EDITED) View.GONE else View.VISIBLE
    row.again.visibility = if (st == LineState.EDITED) View.VISIBLE else View.GONE
    row.again.isEnabled = !busy
    row.note.text = row.error ?: ""
    row.note.visibility = if (st == LineState.FAILED) View.VISIBLE else View.GONE
    // the picture being scored has a thick blue frame
    (row.frame.background as GradientDrawable).setStroke(
      dp(if (row.running) 3f else 1f).toInt(),
      if (row.running) SCORING else FRAME,
    )
    row.details.visibility = if (editing === row) View.VISIBLE else View.GONE
    row.info.text =
      listOfNotNull(
          "OCR conf %.2f".format(row.line.confidence),
          row.ms?.let { "%.0f ms".format(it) },
          if (row.buildMs > 0) "+%.1f s GPU build, once".format(row.buildMs / 1000) else null,
        )
        .joinToString(" · ")
    overlay.invalidate()
  }

  private fun refreshAll() {
    thresholdLabel.text = "Accept at ≥ %.2f ".format(threshold)
    rows.forEach { refreshRow(it) }
    refreshLines()
    val scored = rows.filter { it.prob != null && !it.running }
    val accepted = scored.count { it.prob!! >= threshold }
    if (!check && rows.isNotEmpty() && !busy) {
      showTally(rows.size, null)
      summary.setTextColor(Color.DKGRAY)
      summary.text = "The check is off: every line is accepted as read."
      return
    }
    if (check && busy && rows.isNotEmpty()) {
      // the count so far, one line at a time
      showTally(accepted, scored.size - accepted)
      val at = rows.indexOfFirst { it.running }
      summary.setTextColor(Color.DKGRAY)
      summary.text = if (at >= 0) "Checking line ${at + 1} of ${rows.size}" else "Checking the lines"
      return
    }
    if (scored.isEmpty() && rows.isNotEmpty() && !busy) {
      tally.visibility = View.GONE
      summary.setTextColor(Color.BLACK)
      summary.text =
        if (rows.size == 1) "1 line, not scored (see the line)" else "${rows.size} lines, none scored (see each line)"
    }
    if (scored.isNotEmpty() && !busy) {
      showTally(accepted, scored.size - accepted)
      val times = scored.mapNotNull { it.ms }.sorted()
      val median =
        when {
          times.isEmpty() -> null
          times.size % 2 == 1 -> times[times.size / 2]
          else -> (times[times.size / 2 - 1] + times[times.size / 2]) / 2
        }
      val unscored = rows.size - scored.size
      summary.setTextColor(Color.DKGRAY)
      summary.text =
        listOfNotNull(
            median?.let { "median %.0f ms a line".format(it) },
            if (unscored > 0) "$unscored not scored (see the lines)" else null,
          )
          .joinToString(" · ")
    }
  }

  /** The count under the photo: lines accepted and, with the check on, lines to review. */
  private fun showTally(accepted: Int, review: Int?) {
    val s = SpannableStringBuilder()
    s.append("$accepted accepted", ForegroundColorSpan(ACCEPT), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    if (review != null) {
      s.append(" · ", ForegroundColorSpan(Color.GRAY), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
      s.append("$review to review", ForegroundColorSpan(REVIEW), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    }
    tally.text = s
    tally.visibility = View.VISIBLE
  }

  /**
   * The model's score of a line: a bar as long as the score (the whole bar is 1) filled in the
   * mark's color, with the score and the mark written on it (white on the filled part). Without a
   * score, the whole pill is the mark's color with its word (Accept with the check off, checking…,
   * not scored). Every line's pill has the same length, so the bars compare.
   */
  private inner class ScorePill(context: Context) : View(context) {
    private var value: Double? = null
    private var label: String? = null
    private var color = Color.GRAY
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val words =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = dp(13f)
        typeface = Typeface.DEFAULT_BOLD
      }
    private val outline =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1.5f)
      }
    private val w = PILL_WORDS.maxOf { words.measureText(it) } + dp(14f)
    private val h = words.textSize + dp(10f)

    fun set(v: Double?, l: String?, c: Int) {
      value = v
      label = l
      color = c
      invalidate()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) =
      setMeasuredDimension((w + dp(2f)).toInt(), (h + dp(2f)).toInt())

    override fun onDraw(canvas: Canvas) {
      val l = label ?: return
      val m = dp(1f)
      val r = RectF(m, m, m + w, m + h)
      val rad = h / 2
      val s = value?.let { "%.2f %s".format(it, l) } ?: l
      val end = r.left + w * (value?.toFloat()?.coerceIn(0f, 1f) ?: 1f)
      val baseline = r.top + h / 2 + words.textSize * 0.36f
      fill.color = TRACK
      canvas.drawRoundRect(r, rad, rad, fill)
      canvas.save()
      canvas.clipRect(r.left, r.top, end, r.bottom)
      fill.color = color
      canvas.drawRoundRect(r, rad, rad, fill)
      words.color = Color.WHITE
      canvas.drawText(s, r.left + dp(7f), baseline, words)
      canvas.restore()
      canvas.save()
      canvas.clipRect(end, r.top, r.right, r.bottom)
      words.color = color
      canvas.drawText(s, r.left + dp(7f), baseline, words)
      canvas.restore()
      outline.color = color
      canvas.drawRoundRect(r, rad, rad, outline)
    }
  }

  private enum class LineState { UNCHECKED, WAITING, SCORING, ACCEPT, REVIEW, FAILED, EDITED }

  private fun state(row: Row): LineState =
    when {
      row.running -> LineState.SCORING
      row.stale -> LineState.EDITED
      !check -> LineState.UNCHECKED
      row.error != null -> LineState.FAILED
      row.prob == null -> LineState.WAITING
      row.prob!! >= threshold -> LineState.ACCEPT
      else -> LineState.REVIEW
    }

  /**
   * The lines on the opened photo: a numbered box per line in the color of its mark (green Accept,
   * red Review, blue while it is scored, pale while it waits). A tap on a box scrolls the table to
   * that line.
   */
  private inner class BoxOverlay(context: Context) : View(context) {
    private var p: Photo? = null
    private var shown: List<Row> = emptyList()
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val number =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = dp(11f)
        color = Color.WHITE
      }

    fun set(photo: Photo?, rows: List<Row>) {
      p = photo
      shown = rows
      invalidate()
    }

    private fun boxes(photo: Photo): List<RectF> {
      val s = minOf(width.toFloat() / photo.width, height.toFloat() / photo.height)
      val ox = (width - photo.width * s) / 2
      val oy = (height - photo.height * s) / 2
      return shown.map { row ->
        val b: Rect = row.line.box
        RectF(ox + b.left * s, oy + b.top * s, ox + b.right * s, oy + b.bottom * s)
      }
    }

    override fun onDraw(canvas: Canvas) {
      val photo = p ?: return
      val boxes = boxes(photo)
      val scoring = shown.any { it.running }
      for ((k, row) in shown.withIndex()) {
        val r = boxes[k]
        val st = state(row)
        val color =
          when (st) {
            LineState.UNCHECKED, LineState.ACCEPT -> ACCEPT
            LineState.REVIEW, LineState.FAILED -> REVIEW
            LineState.SCORING -> SCORING
            else -> Color.GRAY
          }
        val boxColor = if (scoring && st == LineState.WAITING) Color.argb(90, 128, 128, 128) else color
        stroke.color = boxColor
        stroke.strokeWidth = if (st == LineState.SCORING) dp(3f) else dp(1.5f)
        canvas.drawRect(r, stroke)
        fill.color = boxColor
        val tag = "${row.index + 1}"
        val tw = number.measureText(tag) + dp(6f)
        canvas.drawRect(r.left - tw, r.top, r.left, r.top + dp(14f), fill)
        canvas.drawText(tag, r.left - tw + dp(3f), r.top + dp(11f), number)
      }
    }

    override fun onTouchEvent(e: MotionEvent): Boolean {
      val photo = p ?: return false
      if (shown.isEmpty()) return false
      if (e.action != MotionEvent.ACTION_UP) return true
      val boxes = boxes(photo)
      val hit =
        shown.indices.firstOrNull {
          val b = boxes[it]
          RectF(b.left - dp(18f), b.top - dp(6f), b.right + dp(6f), b.bottom + dp(6f)).contains(e.x, e.y)
        }
      if (hit != null) {
        performClick()
        showLine(shown[hit])
      }
      return true
    }

    override fun performClick(): Boolean {
      super.performClick()
      return true
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
    private const val PREF_CHECK = "check"
    val ACCEPT = Color.rgb(0x1B, 0x7F, 0x3B)
    val REVIEW = Color.rgb(0xC2, 0x41, 0x0C)
    val SCORING = Color.rgb(0x1E, 0x5A, 0xA8)
    val LABEL = Color.rgb(0x75, 0x75, 0x75)
    /** The gray labels of a row: beside the line's picture and beside the text the recognizer read. */
    private const val IMAGE_LABEL = "Image:"
    private const val TEXT_LABEL = "OCR read:"
    /** The empty part of a score bar. */
    val TRACK = Color.rgb(0xEC, 0xEC, 0xEC)
    /** The frame of a line's picture, and the rule between lines. */
    val FRAME = Color.rgb(0x9E, 0x9E, 0x9E)
    val RULE = Color.rgb(0xE6, 0xE6, 0xE6)
    /** The longest words a score pill holds. */
    private val PILL_WORDS = listOf("0.00 Review", "0.00 Accept", "checking…", "not scored")
    /**
     * The symbol keys: the dollar sign, fractions, superscripts, the degree sign, Greek letters, the
     * times sign, and a key that deletes the character before the cursor.
     */
    private val SYMBOLS = listOf("\$", "½", "¾", "²", "³", "°", "π", "θ", "×", DELETE)
    private const val DELETE = "⌫"
  }
}
