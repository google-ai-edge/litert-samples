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

package com.google.ai.edge.examples.where_to_put

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.OpenableColumns
import android.provider.Settings
import android.text.Editable
import android.text.InputType
import android.text.TextUtils
import android.text.TextWatcher
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import java.io.File
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** How much of the thought the progress line shows while the model thinks (about two short lines). */
private const val LAST_WORDS_CHARS = 140

/** One answer, as the screen showed it (also what the measuring tool records). */
data class AnswerRecord(
  val pictureName: String,
  val request: String,
  val instruction: String,
  val thinking: Boolean,
  /** The model's answer text and, with thinking on, its thought. */
  val text: String,
  val thought: String,
  /** The point read from the answer (null: none could be read), and whether it was marked on the photo. */
  val spot: Spot?,
  val marked: Boolean,
  /** True when the reply ended while the model was still thinking: no answer, nothing marked. */
  val tooLong: Boolean,
  /** The streamed pieces with text (about one per token), and the reply's cap in tokens. */
  val pieces: Int,
  val maxOutputTokens: Int,
  /** From the tap: the first character (thought or answer), the first character of the answer, the mark, the end. */
  val firstTextMs: Long?,
  val answerMs: Long?,
  val markMs: Long?,
  val totalMs: Long,
  val error: String?,
  /** The item drawn on the point (Items.kt) and its width in thousandths of the photo's width; null: a ring only. */
  val item: String? = null,
  val itemWidth: Int? = null,
)

class MainActivity : ComponentActivity() {
  private lateinit var scroll: ScrollView
  private lateinit var status: TextView
  private lateinit var timing: TextView
  private lateinit var photo: MarkedPhotoView
  private lateinit var caption: TextView
  private lateinit var notice: TextView
  private lateinit var exampleSpinner: Spinner
  private lateinit var suggestionsLabel: TextView
  private lateinit var suggestions: LinearLayout
  private lateinit var requestField: EditText
  private lateinit var showButton: Button
  private lateinit var answerView: TextView
  /** While the model thinks: the last words of its thought, small, under the progress line. */
  private lateinit var thinkingView: TextView
  private lateinit var streamLabel: TextView
  private lateinit var streamView: TextView
  private lateinit var thoughtLabel: TextView
  private lateinit var thoughtView: TextView

  private val examples by lazy { Examples.load(this) }
  private var current: Picture? = null
  /** The bundled example on screen, if the picture is one. */
  internal var currentExample: Example? = null
  private var asking = false
  /** The example on screen; a Spinner reports a programmatic selection after layout, later than the call. */
  private var currentExampleId: String? = null
  /** True while the app itself sets the request field (a suggestion, the measuring tool). */
  private var settingRequest = false
  /** Whether the keyboard is open, and where the page was when it opened (null: nowhere to go back to). */
  private var keyboardUp = false
  private var scrollBeforeKeyboard: Int? = null
  /** Where the page was when a finger touched the request field (a tap on a half-shown field scrolls it into view). */
  private var scrollBeforeTap: Int? = null

  /** Set while the measuring tool drives the screen (AutoRun.kt). */
  internal var autoRun: AutoRun? = null

  private val pickPhoto =
    registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
      if (uri != null) loadUri(uri, "Your photo")
    }

  private val takePhoto =
    registerForActivityResult(ActivityResultContracts.TakePicture()) { saved ->
      // The camera app can outlive this activity; the photo is read back from its fixed path.
      val file = cameraFile()
      if (saved && file.length() > 0) loadFile(file, "Camera photo")
      else showNotice("No photo was taken.")
    }

  private val cameraPermission =
    registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) openCamera() else cameraDenied()
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // The page runs under transparent bars on every Android version, with dark status bar icons on the light page (the
    // theme's are white: the clock was white on white).
    enableEdgeToEdge(statusBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT))
    buildScreen()
    handleIntent(intent, fromCreate = true)
    loadModel()
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    handleIntent(intent, fromCreate = false)
  }

  override fun onRestart() {
    super.onRestart()
    // Back from the background (the model file pushed meanwhile, or other apps closed): a load that stopped at the
    // file or memory check is tried again.
    if ((ModelEngine.state as? ModelEngine.State.Unavailable)?.retry == true) loadModel()
  }

  override fun onResume() {
    super.onResume()
    refreshStatus()
  }

  // ---- Inputs -------------------------------------------------------------------------------

  private fun handleIntent(intent: Intent, fromCreate: Boolean) {
    val plan = intent.getStringExtra("plan")
    if (plan != null) {
      // The measuring tool may run on a phone nobody unlocked.
      setShowWhenLocked(true)
      setTurnScreenOn(true)
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
      // `--el need_bytes <n>` raises the memory check's threshold, to show its message on a phone with enough memory;
      // `--es backend cpu` and `--es vision_backend cpu` load the text decoder or the vision encoder on the CPU. All
      // three act before the first load only, so start a fresh process (`am start -S`).
      if (intent.hasExtra("need_bytes")) {
        ModelEngine.neededBytes = intent.getLongExtra("need_bytes", ModelEngine.NEEDED_BYTES)
      }
      // `--ei max_output <tokens>` lowers the reply's cap for the next requests (a reply cut inside the thinking shows
      // the too-long message without a long reply); `--ei max_output 0` puts the app's own cap back.
      if (intent.hasExtra("max_output")) {
        val n = intent.getIntExtra("max_output", 0)
        ModelEngine.maxOutputTokens = if (n > 0) n else ModelEngine.MAX_OUTPUT_TOKENS
      }
      if (ModelEngine.state == ModelEngine.State.NotLoaded || ModelEngine.state is ModelEngine.State.Unavailable) {
        intent.getStringExtra("backend")?.let { ModelEngine.textBackend = if (it == "cpu") "cpu" else "gpu" }
        intent.getStringExtra("vision_backend")?.let { ModelEngine.visionBackend = if (it == "cpu") "cpu" else "gpu" }
      }
      val tag = intent.getStringExtra("report") ?: "report.json"
      autoRun = AutoRun(this, plan, tag).also { run ->
        // A plan sent after the model load has ended does not wait for it.
        val s = ModelEngine.state
        if (s is ModelEngine.State.Ready || s is ModelEngine.State.Unavailable) run.modelSettled()
        lifecycleScope.launch { run.run() }
      }
      return
    }
    val uri: Uri? = when (intent.action) {
      Intent.ACTION_VIEW -> intent.data
      Intent.ACTION_SEND ->
        if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        else @Suppress("DEPRECATION") intent.getParcelableExtra(Intent.EXTRA_STREAM)
      else -> null
    }
    if (uri != null) loadUri(uri, displayName(uri) ?: "Shared picture")
    else if (fromCreate && current == null) selectExample(examples.first().id)
  }

  /** Puts a bundled example on the screen: its photo and its suggested requests; the request field is emptied. */
  fun selectExample(id: String): Boolean {
    val i = examples.indexOfFirst { it.id == id }
    if (i < 0) return false
    val e = examples[i]
    currentExampleId = e.id
    exampleSpinner.setSelection(i + 1)
    setRequest("")
    loadPicture(e.title, e) { Pictures.fromAsset(this, e.file, e.title) }
    return true
  }

  fun loadFile(file: File, name: String) {
    resetExampleSpinner()
    loadPicture(name, null) { Pictures.fromBytes(file.readBytes(), name) }
  }

  private fun loadUri(uri: Uri, name: String) {
    resetExampleSpinner()
    loadPicture(name, null) { Pictures.fromUri(this, uri, name) }
  }

  /** Loads a picture off the main thread; the old picture, mark and answer are cleared first. */
  private fun loadPicture(name: String, example: Example?, read: () -> Picture) {
    current = null
    currentExample = example
    photo.setPhoto(null)
    caption.text = "Reading $name…"
    showSuggestions(example)
    clearAnswer()
    showNotice(null)
    pendingPicture = lifecycleScope.launch {
      try {
        val p = withContext(Dispatchers.IO) { read() }
        current = p
        photo.setPhoto(p.preview)
        caption.text = "${p.name} · ${p.width} × ${p.height}" + (example?.let { " · photo: ${it.credit}" } ?: "")
      } catch (e: PictureException) {
        caption.text = "No picture"
        showNotice(e.message)
      } catch (e: java.io.FileNotFoundException) {
        caption.text = "No picture"
        showNotice("Could not open $name: the file is not there.")
      } catch (e: Exception) {
        // Any other decoder failure is also a picture the app cannot use, not a crash.
        caption.text = "No picture"
        showNotice("Could not read $name as a picture: ${e.message ?: e.javaClass.simpleName}")
      }
      refreshButton()
    }
  }

  /** The picture being read, if any (the measuring tool waits for it). */
  internal var pendingPicture: kotlinx.coroutines.Job? = null

  /** Fills the request field the way a tap on a suggestion does. */
  fun setRequest(text: String) {
    settingRequest = true
    requestField.setText(text)
    requestField.setSelection(requestField.text.length)
    settingRequest = false
    clearAnswer()
  }

  fun takePhotoPressed() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
      PackageManager.PERMISSION_GRANTED) openCamera()
    else cameraPermission.launch(Manifest.permission.CAMERA)
  }

  private fun openCamera() {
    val file = cameraFile().apply { parentFile?.mkdirs(); delete() }
    takePhoto.launch(FileProvider.getUriForFile(this, "$packageName.camera", file))
  }

  private fun cameraDenied() {
    showNotice(
      "Camera permission was denied, so the app cannot take a photo. Allow the camera for this app in Settings " +
        "(Apps > Where to Put It > Permissions), or choose a photo or an example instead."
    )
  }

  private fun cameraFile() = File(cacheDir, "camera/photo.jpg")

  private fun displayName(uri: Uri): String? = try {
    contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
      if (it.moveToFirst()) it.getString(0) else null
    }
  } catch (e: Exception) {
    uri.lastPathSegment
  }

  // ---- The model ----------------------------------------------------------------------------

  private fun loadModel() {
    lifecycleScope.launch {
      refreshStatus()
      ModelEngine.load(applicationContext)
      refreshStatus()
      refreshButton()
      autoRun?.modelSettled()
    }
  }

  /** The Show where button: the same function runs for a finger and for the measuring tool. */
  suspend fun showWhere(): AnswerRecord {
    val tap = SystemClock.elapsedRealtime()
    val p = current
    val request = requestField.text.toString().trim()
    val instruction = Pointing.instruction(request)
    // The item the request names goes on the point as a drawing (a ring alone for an item with none), as wide as the
    // bundled request says it looks in its photo, else at the drawing's own share of the photo's width.
    val drawing = Items.forRequest(request)
    val itemWidth = drawing?.let { d ->
      currentExample?.requests?.firstOrNull { it.text == request }?.itemWidth ?: d.width
    } ?: 0
    val cap = ModelEngine.maxOutputTokens
    val ready = ModelEngine.state is ModelEngine.State.Ready
    val error = when {
      !ready -> (ModelEngine.state as? ModelEngine.State.Unavailable)?.message
        ?: ("The model is not ready yet: " + statusText(ModelEngine.state) + ".")
      p == null -> "Choose a photo, take one, or pick an example first."
      request.isBlank() -> "Type what to put where, for example: Put the mug to the right of the laptop."
      else -> null
    }
    clearAnswer()
    if (error != null) {
      showNotice(error)
      return AnswerRecord(
        p?.name ?: "", request, instruction, ModelEngine.THINKING, "", "", null, false, false, 0, cap,
        null, null, null, 0, error,
      )
    }
    val pic = p!!
    showNotice(null)
    asking = true
    refreshButton()
    revealPhoto()
    // A thinking reply can take a minute: the screen stays on until it ends (unless something else keeps it on).
    val keepOnBefore = (window.attributes.flags and WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) != 0
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    val text = StringBuilder()
    val thought = StringBuilder()
    var pieces = 0
    var firstMs: Long? = null
    var answerMs: Long? = null
    var markMs: Long? = null
    var earlySpot: Spot? = null
    var failure: String? = null
    // The progress line under the photo, twice a second until the mark: what the model is doing and the seconds since
    // the tap. The long texts at the bottom follow at the same pace (setting them for every token would re-lay them
    // out each time).
    val progress = lifecycleScope.launch {
      while (true) {
        if (markMs == null) {
          val phase = when {
            text.isNotEmpty() -> "Writing the answer…"
            thought.isNotEmpty() -> "Thinking…"
            else -> "Looking at the photo…"
          }
          answerView.text = String.format(Locale.US, "%s %d s", phase, (SystemClock.elapsedRealtime() - tap) / 1000)
        }
        showLongTexts(text, thought)
        delay(500)
      }
    }
    try {
      ModelEngine.ask(pic.modelPng, instruction).collect { d ->
        if (d.text.isEmpty() && d.thought.isEmpty()) return@collect
        pieces++
        val now = SystemClock.elapsedRealtime() - tap
        if (firstMs == null) firstMs = now
        if (d.thought.isNotEmpty()) {
          thought.append(d.thought)
          thinkingView.visibility = View.VISIBLE
          thinkingView.text = lastWords(thought)
        }
        if (d.text.isNotEmpty()) {
          if (answerMs == null) answerMs = now
          text.append(d.text)
          // The mark goes on once the answer's point is whole; its label still streams. Only answer text counts: with
          // thinking on, text before any thought (or a "</think>") could be the thinking itself.
          if (earlySpot == null && (!ModelEngine.THINKING || thought.isNotEmpty() || text.contains("</think>"))) {
            Pointing.firstPoint2d(Pointing.answerPart(text.toString()))?.takeIf { it.inPicture }?.let { s ->
              earlySpot = s
              photo.setSpot(s, drawing, itemWidth)
              markMs = now
              answerView.text = markedLine(s)
            }
          }
        }
      }
    } catch (c: CancellationException) {
      // The screen went away: stop here. A cancellation that came from the runtime alone is a failed reply.
      if (!currentCoroutineContext().isActive) throw c
      failure = "The model stopped: ${c.message ?: "cancelled"}"
    } catch (t: Throwable) {
      failure = "The model stopped with an error: ${t.message ?: t.javaClass.simpleName}"
    } finally {
      progress.cancel()
      if (!keepOnBefore) window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
    thinkingView.visibility = View.GONE
    showLongTexts(text, thought)
    var spot: Spot? = null
    if (failure != null && markMs != null) {
      // A reply that failed after its point is no answer either: the mark goes.
      photo.setSpot(null)
      markMs = null
    }
    val answer = Pointing.answerPart(text.toString())
    // A reply that ended while the model was still thinking has no answer: the thought did not close before the reply
    // reached the end of the context (or the cap). Nothing is read from the thought.
    val tooLong = failure == null && ModelEngine.THINKING && answer.isBlank() && thought.isNotBlank()
    if (failure == null) {
      streamLabel.visibility = View.VISIBLE
      streamView.text = text.ifEmpty { "(nothing)" }
      if (tooLong) {
        val end = if (cap >= ModelEngine.CONTEXT_TOKENS) {
          String.format(Locale.US, "the end of the %,d tokens that the photo, the request and the reply share",
            ModelEngine.CONTEXT_TOKENS)
        } else {
          String.format(Locale.US, "its cap of %,d tokens", cap)
        }
        answerView.text = "The answer was too long: the model was still thinking when the reply reached $end. " +
          "There is no point, so nothing is marked. Its thinking is below."
      } else {
        spot = Pointing.read(answer)
        val s = spot
        if (s == null || !s.inPicture) {
          // (Only an answer that wrote "</think>" again after its point could drop the early mark here.)
          photo.setSpot(null)
          markMs = null
        }
        when {
          s == null -> answerView.text = "No point in the model's answer, so nothing is marked. Its answer is below."
          !s.inPicture -> answerView.text = String.format(
            Locale.US, "The model's point (%s, %s) is outside the photo, so nothing is marked. Its answer is below.",
            num(s.x), num(s.y),
          )
          else -> {
            // The mark is already on when the answer's point_2d came while it streamed (the same point: Pointing).
            if (s != earlySpot) {
              photo.setSpot(s, drawing, itemWidth)
              markMs = SystemClock.elapsedRealtime() - tap
            }
            answerView.text = markedLine(s)
          }
        }
      }
    }
    val totalMs = SystemClock.elapsedRealtime() - tap
    timing.text = if (failure != null) {
      String.format(Locale.US, "Stopped %.1f s after the tap", totalMs / 1000.0)
    } else {
      val line = StringBuilder(String.format(Locale.US, "First text %.1f s after the tap", (firstMs ?: totalMs) / 1000.0))
      if (ModelEngine.THINKING) answerMs?.let { line.append(String.format(Locale.US, " · answer at %.1f s", it / 1000.0)) }
      val m = markMs
      line.append(
        when {
          m != null -> String.format(Locale.US, " · mark at %.1f s", m / 1000.0)
          tooLong -> String.format(Locale.US, " · the reply ended at %.1f s, still thinking", totalMs / 1000.0)
          else -> String.format(Locale.US, " · no mark (answer ended at %.1f s)", totalMs / 1000.0)
        }
      )
      line.toString()
    }
    if (failure != null) showNotice(failure)
    asking = false
    refreshButton()
    revealPhoto()
    val record = AnswerRecord(
      pic.name, request, instruction, ModelEngine.THINKING, text.toString(), thought.toString(), spot,
      markMs != null, tooLong, pieces, cap, firstMs, answerMs, markMs, totalMs, failure,
      item = drawing?.name, itemWidth = drawing?.let { itemWidth },
    )
    withContext(Dispatchers.IO) { Records.save(applicationContext, record) }
    return record
  }

  private fun markedLine(s: Spot) = String.format(Locale.US, "Marked: x %s, y %s (of 1000)", num(s.x), num(s.y))

  /** The model's answer and its thought, in full, at the bottom of the screen. */
  private fun showLongTexts(text: CharSequence, thought: CharSequence) {
    if (thought.isNotEmpty() && thoughtView.length() != thought.length) {
      thoughtLabel.visibility = View.VISIBLE
      thoughtView.text = thought.toString()
    }
    if (text.isNotEmpty() && streamView.length() != text.length) {
      streamLabel.visibility = View.VISIBLE
      streamView.text = text.toString()
    }
  }

  /** The last words of the thought, about two short lines, for the progress line while the model thinks. */
  private fun lastWords(thought: CharSequence): String {
    val flat = thought.toString().replace('\n', ' ').trimEnd()
    if (flat.length <= LAST_WORDS_CHARS) return flat
    val cut = flat.substring(flat.length - LAST_WORDS_CHARS)
    return "…" + cut.substring(cut.indexOf(' ') + 1)
  }

  internal fun currentModelPng(): ByteArray? = current?.modelPng

  /** The photo with its mark, for the measuring tool's view dump. */
  internal val photoView: MarkedPhotoView get() = photo

  private fun num(v: Double): String = if (v == Math.floor(v)) v.toLong().toString() else String.format(Locale.US, "%.1f", v)

  // ---- Screen -------------------------------------------------------------------------------

  private fun clearAnswer() {
    photo.setSpot(null)
    answerView.text = ""
    thinkingView.text = ""
    thinkingView.visibility = View.GONE
    streamView.text = ""
    streamLabel.visibility = View.GONE
    thoughtView.text = ""
    thoughtLabel.visibility = View.GONE
    timing.text = ""
  }

  fun showNotice(text: String?) {
    notice.text = text ?: ""
    notice.visibility = if (text == null) View.GONE else View.VISIBLE
  }

  private fun showSuggestions(example: Example?) {
    suggestions.removeAllViews()
    val list = example?.requests ?: emptyList()
    suggestionsLabel.visibility = if (list.isEmpty()) View.GONE else View.VISIBLE
    for (r in list) {
      suggestions.addView(Button(this).apply {
        text = r.text
        isAllCaps = false
        gravity = Gravity.START or Gravity.CENTER_VERTICAL
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
        setOnClickListener { setRequest(r.text) }
      })
    }
  }

  /** Scrolls the status lines and the photo (with its mark) into view: the answer is the mark, not the text below. */
  private fun revealPhoto() {
    scrollBeforeKeyboard = null   // the photo stays in view when the keyboard closes afterwards
    scroll.post { scroll.smoothScrollTo(0, status.top) }
  }

  /**
   * The keyboard scrolls the page to the request field (the page gets shorter); when it closes, the page goes back to
   * where it was before the finger touched the field, so the heading pushed off the top comes back.
   */
  private fun keyboardMoved(up: Boolean) {
    if (up && !keyboardUp) scrollBeforeKeyboard = scrollBeforeTap ?: scroll.scrollY
    if (!up && keyboardUp) scrollBeforeKeyboard?.let { y -> scroll.post { scroll.smoothScrollTo(0, y) } }
    if (!up) scrollBeforeKeyboard = null
    scrollBeforeTap = null
    keyboardUp = up
  }

  private fun refreshButton() {
    showButton.isEnabled = !asking && current != null && ModelEngine.state is ModelEngine.State.Ready
  }

  fun refreshStatus() {
    val airplane = Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
    status.text = "LFM3-VL-2B · ${statusText(ModelEngine.state)} · LiteRT-LM ${BuildConfig.LITERT_LM_VERSION} · " +
      "${Build.MODEL} · " + (if (airplane) "airplane mode on" else "airplane mode off")
    val s = ModelEngine.state
    if (s is ModelEngine.State.Unavailable) {
      showNotice(s.message)
      modelNotice = s.message
    } else if (modelNotice != null) {
      // The load was tried again: its old message goes, unless another message replaced it meanwhile.
      if (notice.text.toString() == modelNotice) showNotice(null)
      modelNotice = null
    }
  }

  /** The model's message that [refreshStatus] put in the notice. */
  private var modelNotice: String? = null

  internal fun statusText(s: ModelEngine.State) = when (s) {
    ModelEngine.State.NotLoaded -> "not loaded"
    ModelEngine.State.Loading -> "loading (${ModelEngine.backendsLabel()})"
    is ModelEngine.State.Ready -> String.format(Locale.US, "%s, ready (loaded in %.1f s)", s.backends, s.loadMs / 1000.0)
    is ModelEngine.State.Unavailable -> "not available"
  }

  internal fun screenLines(): Map<String, String> = mapOf(
    "status" to status.text.toString(),
    "timing" to timing.text.toString(),
    "picture" to caption.text.toString(),
    "notice" to notice.text.toString(),
    "request" to requestField.text.toString(),
    "suggestions" to (0 until suggestions.childCount).joinToString(" | ") { (suggestions.getChildAt(it) as Button).text },
    "answer" to answerView.text.toString(),
    "thinking" to thinkingView.text.toString(),
    "stream" to streamView.text.toString(),
    "thought" to thoughtView.text.toString(),
  )

  private fun resetExampleSpinner() {
    currentExampleId = null
    exampleSpinner.setSelection(0)
  }

  private fun buildScreen() {
    val dp = { v: Int -> TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt() }
    val column = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(dp(16), dp(12), dp(16), dp(24))
      // The page itself takes the first focus. On a phone driven by keys rather than touch the first focus went to the
      // Show where button near the bottom, and the page scrolled down to it, cutting off the heading. No focus
      // highlight: with keys the system would draw one over the whole page.
      isFocusableInTouchMode = true
      defaultFocusHighlightEnabled = false
    }
    fun text(size: Float, bold: Boolean = false, color: Int = Color.rgb(30, 30, 30)) = TextView(this).apply {
      setTextSize(TypedValue.COMPLEX_UNIT_SP, size)
      setTextColor(color)
      if (bold) setTypeface(typeface, Typeface.BOLD)
    }
    fun button(label: String, onClick: () -> Unit) = Button(this).apply {
      text = label
      isAllCaps = false
      setOnClickListener { onClick() }
    }
    val wide = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

    column.addView(text(22f, bold = true).apply { text = getString(R.string.app_name) })
    column.addView(text(14f, color = Color.rgb(70, 70, 70)).apply {
      text = "Photograph a table, a desk or a shelf and say what to put where: the model marks the free spot, " +
        "on this phone."
    })
    status = text(12f, color = Color.rgb(90, 90, 90)).apply { setPadding(0, dp(6), 0, 0) }
    column.addView(status)
    timing = text(12f, color = Color.rgb(90, 90, 90)).apply { setPadding(0, 0, 0, dp(6)) }
    column.addView(timing)

    photo = MarkedPhotoView(this).apply { setBackgroundColor(Color.rgb(238, 238, 238)) }
    column.addView(photo, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(300)))
    caption = text(12f, color = Color.rgb(90, 90, 90))
    column.addView(caption)
    answerView = text(18f, bold = true, color = Color.rgb(16, 64, 128)).apply { setPadding(0, dp(6), 0, dp(2)) }
    column.addView(answerView)
    thinkingView = text(12f, color = Color.rgb(110, 110, 110)).apply {
      visibility = View.GONE
      maxLines = 3
      ellipsize = TextUtils.TruncateAt.END
      setTypeface(typeface, Typeface.ITALIC)
    }
    column.addView(thinkingView)
    notice = text(14f, color = Color.rgb(176, 32, 32)).apply { visibility = View.GONE; setPadding(0, dp(4), 0, dp(4)) }
    column.addView(notice)

    column.addView(LinearLayout(this).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
      addView(button("Take photo") { takePhotoPressed() }, wide)
      addView(button("Choose photo") {
        pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
      }, wide)
    })
    exampleSpinner = Spinner(this).apply {
      adapter = ArrayAdapter(
        this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
        listOf("Examples…") + examples.map { it.title },
      )
      onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
          if (position > 0 && examples[position - 1].id != currentExampleId) {
            selectExample(examples[position - 1].id)
          }
        }
        override fun onNothingSelected(parent: AdapterView<*>?) {}
      }
    }
    column.addView(exampleSpinner)
    suggestionsLabel = text(12f, color = Color.rgb(90, 90, 90)).apply {
      text = "Try one of these, or type your own:"
      setPadding(0, dp(6), 0, 0)
    }
    column.addView(suggestionsLabel)
    suggestions = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    column.addView(suggestions)

    requestField = EditText(this).apply {
      hint = "Put the mug to the right of the laptop."
      inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES or InputType.TYPE_TEXT_FLAG_MULTI_LINE
      addTextChangedListener(object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        // A new request makes the old mark stale.
        override fun afterTextChanged(s: Editable?) { if (!settingRequest && !asking) clearAnswer() }
      })
      // The page's place when the finger comes down, before the tap's focus scrolls a half-shown field into view.
      setOnTouchListener { _, e ->
        if (e.actionMasked == MotionEvent.ACTION_DOWN && !keyboardUp) scrollBeforeTap = scroll.scrollY
        false
      }
    }
    column.addView(requestField)

    showButton = button("Show where") { lifecycleScope.launch { showWhere() } }.apply {
      setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
    }
    column.addView(showButton, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(56)))

    streamLabel = text(12f, color = Color.rgb(90, 90, 90)).apply {
      text = "What the model wrote"; visibility = View.GONE; setPadding(0, dp(10), 0, 0)
    }
    column.addView(streamLabel)
    streamView = text(13f).apply { typeface = Typeface.MONOSPACE }
    column.addView(streamView)
    thoughtLabel = text(12f, color = Color.rgb(90, 90, 90)).apply {
      text = "What the model thought first"; visibility = View.GONE; setPadding(0, dp(10), 0, 0)
    }
    column.addView(thoughtLabel)
    thoughtView = text(12f, color = Color.rgb(90, 90, 90))
    column.addView(thoughtView)

    scroll = ScrollView(this).apply {
      addView(column)
      // With keys rather than touch the first focus can also land on the ScrollView itself, whose focus highlight
      // greys out the whole page until the first touch.
      defaultFocusHighlightEnabled = false
    }
    // The bars' and the keyboard's space comes off a frame around the page instead of padding the page: nothing of the
    // page is drawn under the status bar, and a ScrollView that gets shorter scrolls the focused request field into view
    // by itself, so the words being typed stay above the keyboard (with the keyboard as padding the field stayed under
    // it).
    val root = FrameLayout(this).apply { addView(scroll) }
    ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
      val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
      v.setPadding(bars.left, bars.top, bars.right, maxOf(bars.bottom, ime.bottom))
      keyboardMoved(insets.isVisible(WindowInsetsCompat.Type.ime()))
      WindowInsetsCompat.CONSUMED
    }
    setContentView(root)
    refreshButton()
  }
}
