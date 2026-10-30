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

package com.google.ai.edge.examples.timetable_receipt

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
import android.text.InputType
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.StyleSpan
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import java.io.File
import java.math.BigDecimal
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** One answer, as the screen showed it (also what the measuring tool records). */
data class AnswerRecord(
  val task: Task,
  val pictureName: String,
  val question: String,
  val text: String,
  val answer: String,
  val values: List<String>,
  val firstTextMs: Long?,
  val answerMs: Long,
  val error: String?,
)

class MainActivity : ComponentActivity() {
  private lateinit var scroll: ScrollView
  private lateinit var status: TextView
  private lateinit var picture: ImageView
  private lateinit var pictureCaption: TextView
  private lateinit var notice: TextView
  private lateinit var exampleSpinner: Spinner
  private lateinit var taskGroup: RadioGroup
  private lateinit var afterRow: LinearLayout
  private lateinit var afterField: EditText
  private lateinit var peopleRow: LinearLayout
  private lateinit var peopleField: EditText
  private lateinit var questionField: EditText
  private lateinit var askButton: Button
  private lateinit var answerView: TextView
  private lateinit var streamLabel: TextView
  private lateinit var streamView: TextView
  private lateinit var valuesLabel: TextView
  private lateinit var valuesView: TextView
  private lateinit var timingView: TextView

  private val examples by lazy { Examples.load(this) }
  private var current: Picture? = null
  private var task = Task.NEXT_DEPARTURE
  private var asking = false
  private var userTouched = false
  /** The example on screen; a Spinner reports a programmatic selection after layout, later than the call. */
  private var currentExampleId: String? = null

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
      // `--el need_bytes <n>` raises the memory check's threshold, to show its message on a phone with enough
      // memory; the check runs before the first load only, so start a fresh process (`am start -S`).
      if (intent.hasExtra("need_bytes")) {
        ModelEngine.neededBytes = intent.getLongExtra("need_bytes", ModelEngine.NEEDED_BYTES)
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

  fun selectExample(id: String): Boolean {
    val i = examples.indexOfFirst { it.id == id }
    if (i < 0) return false
    val e = examples[i]
    currentExampleId = e.id
    exampleSpinner.setSelection(i + 1)
    setTask(e.task)
    afterField.setText(e.after)
    peopleField.setText(e.people.toString())
    loadPicture(e.title) { Pictures.fromAsset(this, e.file, e.title) }
    return true
  }

  fun loadFile(file: File, name: String) {
    resetExampleSpinner()
    loadPicture(name) { Pictures.fromBytes(file.readBytes(), name) }
  }

  private fun loadUri(uri: Uri, name: String) {
    resetExampleSpinner()
    loadPicture(name) { Pictures.fromUri(this, uri, name) }
  }

  /** Loads a picture off the main thread; the old picture and answer are cleared first. */
  private fun loadPicture(name: String, read: () -> Picture) {
    current = null
    picture.setImageDrawable(null)
    pictureCaption.text = "Reading $name…"
    clearAnswer()
    showNotice(null)
    pendingPicture = lifecycleScope.launch {
      try {
        val p = withContext(Dispatchers.IO) { read() }
        current = p
        picture.setImageBitmap(p.preview)
        pictureCaption.text = "${p.name} · ${p.width} × ${p.height}"
      } catch (e: PictureException) {
        pictureCaption.text = "No picture"
        showNotice(e.message)
      } catch (e: java.io.FileNotFoundException) {
        pictureCaption.text = "No picture"
        showNotice("Could not open $name: the file is not there.")
      } catch (e: Exception) {
        // Any other decoder failure is also a picture the app cannot use, not a crash.
        pictureCaption.text = "No picture"
        showNotice("Could not read $name as a picture: ${e.message ?: e.javaClass.simpleName}")
      }
      refreshAsk()
    }
  }

  /** The picture being read, if any (the measuring tool waits for it). */
  internal var pendingPicture: kotlinx.coroutines.Job? = null

  fun setTask(t: Task) {
    task = t
    taskGroup.check(TASK_IDS.getValue(t))
    afterRow.visibility = if (t == Task.NEXT_DEPARTURE) View.VISIBLE else View.GONE
    peopleRow.visibility = if (t == Task.SPLIT_BILL) View.VISIBLE else View.GONE
    questionField.visibility = if (t == Task.ASK_ANYTHING) View.VISIBLE else View.GONE
    clearAnswer()
  }

  fun setAfter(v: String) = afterField.setText(v)

  fun setPeople(n: Int) = peopleField.setText(n.toString())

  fun setQuestion(q: String) = questionField.setText(q)

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
      "Camera permission was denied, so the app cannot take a photo. Allow the camera for this " +
        "app in Settings (Apps > Timetable & Receipt > Permissions), or choose a photo or an " +
        "example instead."
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
      refreshAsk()
      autoRun?.modelSettled()
    }
  }

  /** The Ask button: the same function runs for a finger and for the measuring tool. */
  suspend fun ask(): AnswerRecord {
    val tap = SystemClock.elapsedRealtime()
    val p = current
    val question = questionText()
    val ready = ModelEngine.state is ModelEngine.State.Ready
    val error = when {
      !ready -> (ModelEngine.state as? ModelEngine.State.Unavailable)?.message
        ?: ("The model is not ready yet: " + statusText(ModelEngine.state) + ".")
      p == null -> "Choose a photo, take one, or pick an example first."
      task == Task.NEXT_DEPARTURE && Reading.normalizeTime(afterField.text.toString()) == null ->
        "Type the time as HH:MM (for example 09:15)."
      task == Task.SPLIT_BILL && people() == null -> "Type how many people share the bill (1 to 20)."
      task == Task.ASK_ANYTHING && question.isBlank() -> "Type a question about the picture."
      else -> null
    }
    clearAnswer()
    if (error != null) {
      showNotice(error)
      return AnswerRecord(task, p?.name ?: "", question, "", "", emptyList(), null, 0, error)
    }
    val pic = p!!
    showNotice(null)
    asking = true
    userTouched = false
    refreshAsk()
    answerView.text = "Reading the picture…"
    streamLabel.visibility = View.VISIBLE
    val text = StringBuilder()
    var firstMs: Long? = null
    var failure: String? = null
    try {
      ModelEngine.ask(pic.modelPng, task.instruction(question)).collect { delta ->
        if (delta.isEmpty()) return@collect
        text.append(delta)
        streamView.text = text
        if (firstMs == null) firstMs = SystemClock.elapsedRealtime() - tap
        follow()
      }
    } catch (t: Throwable) {
      failure = "The model stopped with an error: ${t.message ?: t.javaClass.simpleName}"
    }
    val (answer, values) = if (failure != null) Pair("", emptyList()) else showResult(text.toString())
    val answerMs = SystemClock.elapsedRealtime() - tap
    timingView.text = String.format(
      Locale.US, "First word %.1f s after the tap · answer at %.1f s · on this phone, offline",
      (firstMs ?: answerMs) / 1000.0, answerMs / 1000.0,
    )
    if (failure != null) showNotice(failure)
    asking = false
    refreshAsk()
    follow()
    val record = AnswerRecord(task, pic.name, question, text.toString(), answer, values, firstMs, answerMs, failure)
    withContext(Dispatchers.IO) { Records.save(applicationContext, record) }
    return record
  }

  internal fun currentModelPng(): ByteArray? = current?.modelPng

  /** Turns the model's text into the answer and the table of values it rests on. */
  private fun showResult(text: String): Pair<String, List<String>> {
    when (task) {
      Task.NEXT_DEPARTURE -> {
        val after = Reading.normalizeTime(afterField.text.toString())!!
        val r = Reading.timetable(text, after)
        if (r.times.isEmpty()) {
          answerView.text = if (r.saidNone) "No timetable found in this picture."
          else "No departure times could be read from this picture."
          return Pair(answerView.text.toString(), emptyList())
        }
        answerView.text = if (r.next != null) "Next departure after $after:  ${r.next}"
        else "No departure after $after among the times read."
        val table = SpannableStringBuilder()
        r.times.forEachIndexed { i, t ->
          val line = String.format(Locale.US, "%2d.  %s", i + 1, t)
          val start = table.length
          table.append(line)
          if (t == r.next) {
            table.append("   ← next")
            table.setSpan(StyleSpan(Typeface.BOLD), start, table.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
          }
          table.append('\n')
        }
        showValues("Departure times read (${r.times.size})", table)
        return Pair(answerView.text.toString(), r.times)
      }
      Task.SPLIT_BILL -> {
        val r = Reading.receipt(text, people() ?: 1)
        if (r.lines.isEmpty() || r.total == null) {
          answerView.text = if (r.saidNone) "No receipt found in this picture."
          else "No prices could be read from this picture."
          return Pair(answerView.text.toString(), emptyList())
        }
        answerView.text = String.format(
          Locale.US, "Total %s  ·  %d people  ·  %s each",
          money(r.total), r.people, money(r.perPerson!!),
        )
        val table = SpannableStringBuilder()
        for (l in r.lines) {
          val start = table.length
          table.append(String.format(Locale.US, "%-22s %8s\n", l.name.take(22), l.amount.toPlainString()))
          if (l.role == ReceiptLine.Role.TOTAL) {
            table.setSpan(StyleSpan(Typeface.BOLD), start, table.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
          }
        }
        table.append('\n')
        r.subtotal?.let { table.append(agreement("Items add up to", r.itemsSum, "subtotal", it)).append('\n') }
        if (r.subtotal != null && r.tax != null) {
          table.append(agreement("Subtotal + tax =", r.subtotal + r.tax, "total", r.total)).append('\n')
        }
        table.append("Total taken from ${r.totalSource}.")
        showValues("Lines read (${r.lines.size})", table)
        return Pair(answerView.text.toString(), r.lines.map { it.amount.toPlainString() })
      }
      Task.ASK_ANYTHING -> {
        answerView.text = text.trim()
        return Pair(text.trim(), emptyList())
      }
    }
  }

  private fun agreement(what: String, got: BigDecimal, name: String, expected: BigDecimal): String =
    if (got.compareTo(expected) == 0) "$what ${got.toPlainString()} = the $name ✓"
    else "$what ${got.toPlainString()}, the $name reads ${expected.toPlainString()} (check it)"

  private fun money(v: BigDecimal) = "$" + v.toPlainString()

  private fun people(): Int? = peopleField.text.toString().trim().toIntOrNull()?.takeIf { it in 1..20 }

  private fun questionText() = questionField.text.toString()

  // ---- Screen -------------------------------------------------------------------------------

  private fun clearAnswer() {
    answerView.text = ""
    streamView.text = ""
    streamLabel.visibility = View.GONE
    valuesLabel.visibility = View.GONE
    valuesView.text = ""
    timingView.text = ""
  }

  private fun showValues(label: String, table: CharSequence) {
    valuesLabel.text = label
    valuesLabel.visibility = View.VISIBLE
    valuesView.text = table
  }

  fun showNotice(text: String?) {
    notice.text = text ?: ""
    notice.visibility = if (text == null) View.GONE else View.VISIBLE
  }

  /** Keeps the end of the answer in view while it streams, until the user touches the screen. */
  private fun follow() {
    if (userTouched) return
    scroll.post { scroll.smoothScrollTo(0, (timingView.bottom - scroll.height).coerceAtLeast(0)) }
  }

  private fun refreshAsk() {
    askButton.isEnabled = !asking && current != null && ModelEngine.state is ModelEngine.State.Ready
  }

  fun refreshStatus() {
    val airplane = Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
    status.text = "LFM3-VL-2B · ${statusText(ModelEngine.state)} · " +
      (if (airplane) "airplane mode on" else "airplane mode off") +
      " · ${Build.MANUFACTURER} ${Build.MODEL}"
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
    ModelEngine.State.Loading -> "loading on the GPU (the first start takes about a minute)"
    is ModelEngine.State.Ready -> String.format(Locale.US, "GPU, ready (loaded in %.1f s)", s.loadMs / 1000.0)
    is ModelEngine.State.Unavailable -> "not available"
  }

  internal fun screenLines(): Map<String, String> = mapOf(
    "status" to status.text.toString(),
    "picture" to pictureCaption.text.toString(),
    "notice" to notice.text.toString(),
    "task" to task.name,
    "answer" to answerView.text.toString(),
    "stream" to streamView.text.toString(),
    "values" to valuesView.text.toString(),
    "timing" to timingView.text.toString(),
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
    fun row() = LinearLayout(this).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
    }
    val wide = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

    column.addView(text(22f, bold = true).apply { text = getString(R.string.app_name) })
    column.addView(text(14f, color = Color.rgb(70, 70, 70)).apply {
      text = "Photograph a timetable or a receipt: the model reads it on this phone, offline, " +
        "and the app works out the answer."
    })
    status = text(12f, color = Color.rgb(90, 90, 90)).apply { setPadding(0, dp(6), 0, dp(6)) }
    column.addView(status)

    picture = ImageView(this).apply {
      scaleType = ImageView.ScaleType.FIT_CENTER
      setBackgroundColor(Color.rgb(238, 238, 238))
      adjustViewBounds = false
    }
    column.addView(picture, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(260)))
    pictureCaption = text(12f, color = Color.rgb(90, 90, 90))
    column.addView(pictureCaption)
    notice = text(14f, color = Color.rgb(176, 32, 32)).apply { visibility = View.GONE; setPadding(0, dp(4), 0, dp(4)) }
    column.addView(notice)

    column.addView(row().apply {
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

    taskGroup = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
    for (t in Task.entries) {
      taskGroup.addView(RadioButton(this).apply {
        id = TASK_IDS.getValue(t)
        text = t.label
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setOnClickListener { setTask(t) }
      })
    }
    column.addView(taskGroup)

    afterField = EditText(this).apply { inputType = InputType.TYPE_CLASS_DATETIME or InputType.TYPE_DATETIME_VARIATION_TIME; hint = "09:15" }
    afterRow = row().apply {
      addView(text(15f).apply { text = "Leaving after (HH:MM)  " })
      addView(afterField, wide)
    }
    column.addView(afterRow)
    peopleField = EditText(this).apply { inputType = InputType.TYPE_CLASS_NUMBER; hint = "2" }
    peopleRow = row().apply {
      addView(text(15f).apply { text = "People sharing the bill  " })
      addView(peopleField, wide)
    }
    column.addView(peopleRow)
    questionField = EditText(this).apply {
      hint = "Ask anything about the picture"
      inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
    }
    column.addView(questionField)

    askButton = button("Ask") { lifecycleScope.launch { ask() } }.apply {
      setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
    }
    column.addView(askButton, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(56)))

    answerView = text(24f, bold = true, color = Color.rgb(16, 64, 128)).apply { setPadding(0, dp(10), 0, dp(6)) }
    column.addView(answerView)
    streamLabel = text(12f, color = Color.rgb(90, 90, 90)).apply { text = "What the model writes"; visibility = View.GONE }
    column.addView(streamLabel)
    streamView = text(14f).apply { typeface = Typeface.MONOSPACE }
    column.addView(streamView)
    valuesLabel = text(12f, color = Color.rgb(90, 90, 90)).apply { setPadding(0, dp(10), 0, 0); visibility = View.GONE }
    column.addView(valuesLabel)
    valuesView = text(14f).apply { typeface = Typeface.MONOSPACE }
    column.addView(valuesView)
    timingView = text(12f, color = Color.rgb(90, 90, 90)).apply { setPadding(0, dp(10), 0, 0) }
    column.addView(timingView)

    scroll = ScrollView(this).apply {
      addView(column)
      setOnTouchListener { _, _ -> userTouched = true; false }
    }
    ViewCompat.setOnApplyWindowInsetsListener(scroll) { v, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
      v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
      insets
    }
    setContentView(scroll)
    setTask(Task.NEXT_DEPARTURE)
    refreshAsk()
  }

  companion object {
    private val TASK_IDS = mapOf(
      Task.NEXT_DEPARTURE to View.generateViewId(),
      Task.SPLIT_BILL to View.generateViewId(),
      Task.ASK_ANYTHING to View.generateViewId(),
    )
  }
}
