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

package com.google.ai.edge.examples.photo_voice_chat

import android.Manifest
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.provider.Settings
import android.text.InputType
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.StyleSpan
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONArray
import org.json.JSONObject

/**
 * One screen: a photo (the camera, the phone's photos or an example), then questions about it, spoken (hold the button
 * while you speak; a short tap starts and a second tap ends) or typed. Each answer's text appears as it is written and
 * its speech plays as it is decoded; the next question goes on from the answers before it. Every answer leaves a
 * record in files/runs/ (the timings, the text, the speech). Started with a plan (`--es plan`), the same screen is
 * driven by [AutoRun]: it presses the same buttons' code (the measuring tool, README.md).
 */
class MainActivity : Activity() {
  private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "litert").apply { isDaemon = true } }
  private val io = Executors.newSingleThreadExecutor { r -> Thread(r, "io").apply { isDaemon = true } }
  private var engine: Lfm3OmniEngine? = null
  private var conversation: Conversation? = null
  private var loading = false
  private var whenReady = ArrayList<() -> Unit>()
  @Volatile private var stopFlag = false
  @Volatile private var busy = false
  @Volatile private var photoReady = false
  private var photo: PhotoInput? = null
  private var photoSource: JSONObject = JSONObject()
  private var example: ExamplePhoto? = null
  private var recorder: AudioIO.Recorder? = null
  private var recordDownNs = 0L
  private var tapToStop = false
  private var questionPlayer: AudioIO.Player? = null
  private var answerPlayer: AudioIO.Player? = null
  private var runCount = 0
  private lateinit var examples: List<ExamplePhoto>
  private var autoRun: AutoRun? = null
  /** Text written after each question; none by default: the photo and the question alone, as in the model card's code
   *  example (README.md). The measuring tool can set it. */
  var instruction = DEFAULT_INSTRUCTION

  private lateinit var deviceLine: TextView
  private lateinit var engineLine: TextView
  private lateinit var scroller: ScrollView
  private lateinit var content: LinearLayout
  /** While an answer comes in, the conversation follows its end; a finger on it stops that. */
  private var followAnswer = false
  private lateinit var photoView: ImageView
  private lateinit var photoStatus: TextView
  private lateinit var notice: TextView
  private lateinit var turnsView: LinearLayout
  private lateinit var questionsRow: LinearLayout
  private lateinit var askButton: Button
  /** One pill beside the button: ● LISTENING (red), ● ASKING (blue, an example question playing), ● SPEAKING (green). */
  private lateinit var pill: TextView
  private lateinit var typed: EditText
  private lateinit var typedAsk: Button
  private lateinit var newButton: Button
  private lateinit var stopButton: Button
  private lateinit var heartbeat: View
  private val turnTexts = ArrayList<Pair<String, TextView>>()

  private val airplaneReceiver = object : BroadcastReceiver() {
    override fun onReceive(c: Context?, i: Intent?) = showDevice()
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // A shared test phone nobody unlocked runs the measuring tool too: show over the keyguard and keep the screen (and
    // the top-app CPU set) while an answer runs.
    if (Build.VERSION.SDK_INT >= 27) {
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    examples = ExamplePhoto.list(this)
    buildUi()
    val airplane = IntentFilter(Intent.ACTION_AIRPLANE_MODE_CHANGED)
    if (Build.VERSION.SDK_INT >= 33) registerReceiver(airplaneReceiver, airplane, Context.RECEIVER_NOT_EXPORTED)
    else registerReceiver(airplaneReceiver, airplane)
    // a photo taken while the system stopped this app behind the camera comes back to a new activity: its file is at
    // a fixed path, so the result still finds it
    if (savedInstanceState == null && intent.getStringExtra("plan") == null && !isPhotoIntent(intent)) selectExample(examples.first())
    handle(intent)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    handle(intent)
  }

  override fun onRestart() {
    super.onRestart()
    // back from the background, e.g. after install_to_device.sh pushed the files the first start named: a load that
    // stopped at the file or memory check runs again
    if (engine == null && !loading) loadEngine()
  }

  override fun onResume() {
    super.onResume()
    showDevice()
  }

  private fun isPhotoIntent(i: Intent) = i.action == Intent.ACTION_VIEW || i.action == Intent.ACTION_SEND

  private fun handle(intent: Intent) {
    val plan = intent.getStringExtra("plan")
    when {
      plan != null -> {
        val a = AutoRun(this, plan, intent.getStringExtra("report") ?: "autorun.json", intent.getBooleanExtra("keep_frames", false))
        autoRun = a
        afterLoad { a.start() }
      }
      intent.action == Intent.ACTION_VIEW && intent.data != null -> openPhotoUri(intent.data!!)
      intent.action == Intent.ACTION_SEND -> {
        @Suppress("DEPRECATION")
        val uri = if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        else intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
        if (uri != null) openPhotoUri(uri)
      }
    }
    if (engine == null && !loading) loadEngine()
  }

  // ---- the screen ------------------------------------------------------------------------------------------------

  private fun buildUi() {
    val pad = dp(12)
    val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.WHITE) }
    ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
      v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
      insets
    }
    setContentView(root)

    val header = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, pad, pad, dp(4)) }
    header.addView(text("Photo voice chat", 21f, Color.BLACK, bold = true))
    header.addView(text("Ask about a photo out loud and hear the answer. LFM3-Omni-2B runs on this phone (LiteRT " +
      "${BuildConfig.LITERT_VERSION}); the app has no internet permission, so the photo and your voice stay on the phone.", 13f, Color.DKGRAY))
    deviceLine = text("", 12f, Color.DKGRAY).apply { setPadding(0, dp(4), 0, 0) }
    header.addView(deviceLine)
    engineLine = text("", 12f, Color.DKGRAY)
    header.addView(engineLine)
    root.addView(header)

    content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, 0, pad, pad) }
    scroller = ScrollView(this).apply { addView(content) }
    scroller.setOnTouchListener { _, ev ->
      if (ev.actionMasked == MotionEvent.ACTION_DOWN) followAnswer = false
      false
    }
    content.viewTreeObserver.addOnGlobalLayoutListener { if (followAnswer) scrollToEnd() }
    root.addView(scroller, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

    val photoRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    photoView = ImageView(this).apply {
      scaleType = ImageView.ScaleType.FIT_CENTER
      setBackgroundColor(Color.rgb(238, 238, 238))
      contentDescription = "The photo the model looks at"
    }
    photoRow.addView(photoView, LinearLayout.LayoutParams(dp(196), dp(196)))
    val photoButtons = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(10), 0, 0, 0) }
    photoButtons.addView(button("Camera") { takePhoto() })
    photoButtons.addView(button("Choose photo") { choosePhoto() })
    photoButtons.addView(text("Examples", 12f, Color.GRAY, bold = true).apply { setPadding(0, dp(6), 0, 0) })
    for (e in examples) photoButtons.addView(button(e.label) { selectExample(e) })
    photoRow.addView(photoButtons)
    content.addView(photoRow)
    photoStatus = text("", 13f, Color.DKGRAY).apply { setPadding(0, dp(4), 0, 0) }
    content.addView(photoStatus)
    notice = text("", 14f, Color.rgb(183, 28, 28)).apply { visibility = View.GONE; setPadding(0, dp(4), 0, dp(4)) }
    content.addView(notice)
    turnsView = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    content.addView(turnsView)
    // the measuring tool's recordings: a still screen gives no video frames, so one pixel keeps changing
    heartbeat = View(this).apply { layoutParams = LinearLayout.LayoutParams(dp(1), dp(1)); visibility = View.INVISIBLE }
    content.addView(heartbeat)

    val bar = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(pad, dp(6), pad, dp(6))
      setBackgroundColor(Color.rgb(245, 245, 245))
    }
    // the example questions one per row: each one whole on the screen (a row cut at the edge hid the last one)
    questionsRow = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    bar.addView(questionsRow)
    val askRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
    askButton = Button(this).apply {
      text = "Hold to ask"
      isAllCaps = false
      textSize = 17f
      setOnTouchListener { v, ev ->
        when (ev.actionMasked) {
          MotionEvent.ACTION_DOWN -> micDown()
          MotionEvent.ACTION_UP -> { micUp(); v.performClick() }
          MotionEvent.ACTION_CANCEL -> micUp()
        }
        true
      }
    }
    askRow.addView(askButton, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
    pill = text("", 12f, Color.WHITE, bold = true).apply {
      gravity = Gravity.CENTER
      minWidth = dp(132)
      setPadding(dp(8), dp(3), dp(8), dp(3))
      visibility = View.INVISIBLE
    }
    askRow.addView(pill, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { leftMargin = dp(8) })
    bar.addView(askRow)
    val typedRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
    typed = EditText(this).apply {
      hint = "Or type a question"
      textSize = 15f
      inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
      imeOptions = EditorInfo.IME_ACTION_SEND
      setOnEditorActionListener { _, id, _ -> if (id == EditorInfo.IME_ACTION_SEND) { askTyped(null); true } else false }
    }
    typedRow.addView(typed, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
    typedAsk = button("Ask") { askTyped(null) }
    typedRow.addView(typedAsk)
    bar.addView(typedRow)
    val endRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    newButton = button("New conversation") { newConversation(null) }
    stopButton = button("Stop") { stopAll() }.apply { isEnabled = false }
    endRow.addView(newButton)
    endRow.addView(stopButton)
    bar.addView(endRow)
    root.addView(bar)
    setAskEnabled()
  }

  private fun text(s: String, size: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
    text = s
    textSize = size
    setTextColor(color)
    if (bold) setTypeface(typeface, Typeface.BOLD)
  }

  private fun button(s: String, onClick: () -> Unit) = Button(this).apply {
    text = s
    isAllCaps = false
    setOnClickListener { onClick() }
  }

  /** The pill's state: null = hidden (its place kept, so the button does not move). UI thread. */
  private fun setPill(label: String?, fill: Int = Color.GRAY) {
    if (label == null) {
      pill.visibility = View.INVISIBLE
      return
    }
    pill.text = label
    pill.background = GradientDrawable().apply { cornerRadius = dp(12).toFloat(); setColor(fill) }
    pill.visibility = View.VISIBLE
  }

  private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

  fun airplaneMode(): Boolean = Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1

  private fun showDevice() {
    val soc = if (Build.VERSION.SDK_INT >= 31) " (${Build.SOC_MODEL})" else ""
    runOnUiThread { deviceLine.text = "${Build.MODEL}$soc · airplane mode ${if (airplaneMode()) "on" else "off"}" }
  }

  private fun setEngineLine(s: String) = runOnUiThread { engineLine.text = s }

  private fun setNotice(s: String?) = runOnUiThread {
    notice.text = s ?: ""
    notice.visibility = if (s.isNullOrEmpty()) View.GONE else View.VISIBLE
  }

  private fun setAskEnabled() = runOnUiThread {
    val ready = engine != null && photoReady && !busy
    askButton.isEnabled = ready || recorder?.isRunning == true
    typedAsk.isEnabled = ready
    newButton.isEnabled = ready
    for (i in 0 until questionsRow.childCount) questionsRow.getChildAt(i).isEnabled = ready
  }

  private fun scrollToEnd() {
    val end = content.height - (scroller.height - scroller.paddingTop - scroller.paddingBottom)
    if (end > scroller.scrollY) scroller.smoothScrollTo(0, end)
  }

  private fun clearTurns() = runOnUiThread {
    turnsView.removeAllViews()
    turnTexts.clear()
    followAnswer = false
  }

  private class TurnViews(val answer: TextView, val metrics: TextView)

  private fun addTurn(question: String): TurnViews {
    // the question in gray: the blue and the green are the pills' own colours (a recording finds the pills by them)
    val q = text(question, 14f, Color.rgb(97, 97, 97), bold = true).apply { setPadding(0, dp(10), 0, dp(2)) }
    val a = text("", 18f, Color.BLACK).apply { setTextIsSelectable(true) }
    val m = text("", 12f, Color.DKGRAY).apply { typeface = Typeface.MONOSPACE }
    turnsView.addView(q)
    turnsView.addView(a)
    turnsView.addView(m)
    turnTexts.add(question to a)
    return TurnViews(a, m)
  }

  // ---- the photo -------------------------------------------------------------------------------------------------

  fun selectExample(e: ExamplePhoto, done: ((Boolean) -> Unit)? = null) {
    if (busy) {
      setNotice("Wait for the answer to finish, or tap Stop.")
      done?.invoke(false)
      return
    }
    val p = try {
      e.load(this)
    } catch (t: Exception) {
      setNotice(t.message)
      done?.invoke(false)
      return
    }
    usePhoto(p, JSONObject().put("kind", "example").put("id", e.id).put("label", e.label), e, done)
  }

  private fun choosePhoto() {
    if (busy) return
    val i = if (Build.VERSION.SDK_INT >= 33) Intent(MediaStore.ACTION_PICK_IMAGES)
    else Intent(Intent.ACTION_OPEN_DOCUMENT).apply { addCategory(Intent.CATEGORY_OPENABLE); type = "image/*" }
    startActivityForResult(i, PICK_PHOTO)
  }

  /** The system camera app, writing to a fixed file (the app needs no camera permission for this). */
  fun takePhoto() {
    if (busy) return
    val f = cameraFile().apply { parentFile?.mkdirs(); delete() }
    val uri = FileProvider.getUriForFile(this, "$packageName.files", f)
    val i = Intent(MediaStore.ACTION_IMAGE_CAPTURE).putExtra(MediaStore.EXTRA_OUTPUT, uri)
      .addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION)
    try {
      startActivityForResult(i, CAMERA)
    } catch (e: ActivityNotFoundException) {
      setNotice("This phone has no camera app to take a photo with. Choose a photo instead.")
    }
  }

  private fun cameraFile() = File(cacheDir, "camera/photo.jpg")

  @Deprecated("Activity result API of the platform")
  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    when (requestCode) {
      PICK_PHOTO -> if (resultCode == RESULT_OK) data?.data?.let { openPhotoUri(it) }
      CAMERA -> {
        val f = cameraFile()
        if (resultCode == RESULT_OK && f.isFile && f.length() > 0) openPhotoUri(Uri.fromFile(f), name = "Camera photo")
        else {
          setNotice("No photo was taken (the camera was closed without a picture)." +
            if (photo != null) " The photo before it is still in use." else "")
          autoRun?.onCameraResult(false)
        }
      }
    }
  }

  /** A picked, shared, opened or taken photo: decoded off the UI thread; [done] (the measuring tool) gets whether it is usable. */
  fun openPhotoUri(uri: Uri, done: ((Boolean) -> Unit)? = null, name: String? = null) {
    if (busy) {
      setNotice("Wait for the answer to finish, or tap Stop.")
      done?.invoke(false)
      return
    }
    val label = name ?: displayName(uri)
    setNotice(null)
    runOnUiThread { photoStatus.text = "Reading $label…" }
    io.execute {
      try {
        val p = Photos.fromUri(this, uri, label)
        runOnUiThread { usePhoto(p, JSONObject().put("kind", "file").put("label", label).put("uri", uri.toString()), null, done) }
        autoRun?.onCameraResult(true)
      } catch (e: UnusableImage) {
        // a photo that cannot be read leaves no photo: the questions must not go to the one before it
        runOnUiThread {
          photo = null
          photoReady = false
          example = null
          photoView.setImageDrawable(null)
          photoStatus.text = "No photo"
          showQuestions(null)
          clearTurns()
          setAskEnabled()
        }
        setNotice(e.message)
        done?.invoke(false)
        autoRun?.onCameraResult(false)
      }
    }
  }

  /** A new photo = a new conversation: the photo goes through the vision encoder and is prefilled with the system turn. */
  private fun usePhoto(p: PhotoInput, source: JSONObject, e: ExamplePhoto?, done: ((Boolean) -> Unit)?) {
    stopQuestionPlayback()
    photo = p
    photoSource = source
    example = e
    photoReady = false
    photoView.setImageBitmap(p.square)
    photoView.contentDescription = "Photo: ${p.name}"
    photoStatus.text = "Looking at the photo…"
    showQuestions(e)
    clearTurns()
    setNotice(null)
    setAskEnabled()
    afterLoad {
      val c = conversation ?: return@afterLoad
      busy = true
      setAskEnabled()
      worker.execute {
        val ok = try {
          val t = c.setPhoto(p)
          photoSource.put("prepare", t)
          runOnUiThread {
            if (photo === p) {
              photoReady = true
              photoStatus.text = "%s · ready, ask about it (looked at it in %.1f s)".format(p.name,
                (t.optDouble("vision_ms") + t.optDouble("start_prefill_ms")) / 1000)
            }
          }
          true
        } catch (t: Throwable) {
          Log.e(Lfm3OmniEngine.TAG, "photo failed", t)
          setNotice("The photo could not be read by the model: ${t.message}")
          false
        }
        busy = false
        setAskEnabled()
        done?.invoke(ok)
      }
    }
  }

  private fun showQuestions(e: ExamplePhoto?) = runOnUiThread {
    questionsRow.removeAllViews()
    if (e == null) return@runOnUiThread
    for (q in e.questions) questionsRow.addView(Button(this).apply {
      text = "“${q.text}”"
      isAllCaps = false
      textSize = 14f
      gravity = Gravity.START or Gravity.CENTER_VERTICAL
      minHeight = 0
      minimumHeight = 0
      setPadding(dp(12), dp(4), dp(12), dp(4))
      setOnClickListener { askExample(q, null) }
    }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
    setAskEnabled()
  }

  private fun displayName(uri: Uri): String {
    if (uri.scheme == "content") runCatching {
      contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
        if (c.moveToFirst()) return c.getString(0)
      }
    }
    return uri.lastPathSegment?.substringAfterLast('/') ?: "the photo"
  }

  // ---- the questions ---------------------------------------------------------------------------------------------

  /** The options the measuring tool can set for one answer; a finger's answer uses the defaults. */
  class AskOptions(
    val seed: Long = 0,
    val history: String = "state",
    val sampling: Sampling = Sampling(),
    val fixedOrder: Boolean = false,
    val dump: String? = null,
    val id: String? = null,
  )

  private fun refusal(): String? = when {
    busy -> "An answer is still running."
    isRecording() -> "Let go of the button first."
    engine == null -> engineLine.text.toString().ifEmpty { "The model is not loaded yet." }
    photo == null -> "Pick a photo first (the camera, Choose photo or an example)."
    !photoReady -> "The model is still looking at the photo."
    else -> null
  }

  /** An example question: its recording plays (as if spoken), and when it ends the question goes to the model. */
  fun askExample(q: ExampleQuestion, done: ((JSONObject?) -> Unit)?, opts: AskOptions = AskOptions()) {
    val clip = try {
      q.load(this)
    } catch (e: Exception) {
      setNotice(e.message)
      done?.invoke(null)
      return
    }
    askRecording(clip, "“${q.text}”", JSONObject().put("kind", "example").put("id", q.id).put("text", q.text), done, opts)
  }

  /** A recording played as the question, then asked (the example questions, and the measuring tool's question files). */
  fun askRecording(clip: Clip, label: String, source: JSONObject, done: ((JSONObject?) -> Unit)?, opts: AskOptions = AskOptions()) {
    refusal()?.let {
      setNotice(it)
      done?.invoke(null)
      return
    }
    setNotice(null)
    busy = true
    setAskEnabled()
    stopButton.isEnabled = true
    val p = AudioIO.Player(16000, onSound = { ns ->
      autoRun?.onQuestionSound(ns)
      runOnUiThread { setPill("● ASKING", ASKING) }
    }, onEnd = { pl ->
      runOnUiThread {
        setPill(null)
        if (questionPlayer === pl) questionPlayer = null
        busy = false
        if (pl.endNs > 0 && !stopFlag) {
          answer(Conversation.Question(clip, "", label), pl.endNs, source.put("seconds", clip.seconds), done, opts)
        } else {
          setAskEnabled()
          stopButton.isEnabled = false
          done?.invoke(null)
        }
      }
    })
    stopFlag = false
    questionPlayer = p
    p.play(clip.pcm16k)
    p.close()
  }

  /** A question recording from a file (the measuring tool): the microphone's checks, then played and asked as an example. */
  fun askWavFile(f: File, label: String?, done: ((JSONObject?) -> Unit)?, opts: AskOptions = AskOptions()) {
    val clip = try {
      AudioIO.check("the recording", AudioIO.readWav16k(f))
    } catch (e: Exception) {
      setNotice(e.message)
      done?.invoke(null)
      return
    }
    askRecording(clip, label ?: "Spoken question (%.1f s)".format(clip.seconds), JSONObject().put("kind", "file").put("file", f.name), done, opts)
  }

  /** The typed question (the Ask button), or [text] from the measuring tool. */
  fun askTyped(text: String?, done: ((JSONObject?) -> Unit)? = null, opts: AskOptions = AskOptions()) {
    val t = (text ?: typed.text.toString()).trim()
    if (t.isEmpty()) {
      setNotice("Type a question first, or hold the button and ask it out loud.")
      done?.invoke(null)
      return
    }
    if (refusal() != null) {
      setNotice(refusal())
      done?.invoke(null)
      return
    }
    runOnUiThread { typed.setText("") }
    answer(Conversation.Question(null, t, "“$t”"), System.nanoTime(), JSONObject().put("kind", "typed").put("text", t), done, opts)
  }

  /** The button pressed: start listening (or, in tap mode, stop and ask). */
  fun micDown() {
    val r = recorder
    if (r != null && r.isRunning) {
      if (tapToStop) finishRecording(r.stop())
      return
    }
    refusal()?.let {
      setNotice(it)
      return
    }
    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), RECORD)
      return
    }
    startRecording()
  }

  /** The button let go: a hold ends the question; a short tap keeps listening until the next tap. */
  fun micUp() {
    val r = recorder ?: return
    if (!r.isRunning || tapToStop) return
    if (System.nanoTime() - recordDownNs < TAP_NS) {
      tapToStop = true
      askButton.text = "Tap to finish the question"
      return
    }
    finishRecording(r.stop())
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode != RECORD) return
    if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
      setNotice("The microphone is on: hold the button while you ask.")
    } else {
      setNotice("Microphone permission was denied, so the app cannot hear a question. Allow the microphone for Photo voice " +
        "chat in Settings (Apps > Photo voice chat > Permissions), or type the question instead.")
      autoRun?.onRecordDenied()
    }
  }

  private fun startRecording() {
    setNotice(null)
    stopQuestionPlayback()
    val rec = AudioIO.Recorder { s, limit ->
      runOnUiThread {
        if (limit) recorder?.let { finishRecording(it.stop()) }
        else setPill("● LISTENING %d:%02d".format(s.toInt() / 60, s.toInt() % 60), LISTENING)
      }
    }
    try {
      rec.start()
    } catch (e: Exception) {
      setNotice("Could not start the microphone: ${e.message}")
      return
    }
    recorder = rec
    recordDownNs = System.nanoTime()
    tapToStop = false
    setPill("● LISTENING 0:00", LISTENING)
    askButton.text = "Let go to ask"
    setAskEnabled()
  }

  private fun finishRecording(pcm: FloatArray) {
    val endNs = System.nanoTime()
    recorder = null
    tapToStop = false
    askButton.text = "Hold to ask"
    setPill(null)
    try {
      val clip = AudioIO.check("the recording", pcm)
      answer(Conversation.Question(clip, "", "Spoken question (%.1f s)".format(clip.seconds)), endNs,
        JSONObject().put("kind", "recording").put("seconds", clip.seconds), null, AskOptions())
    } catch (e: UnusableAudio) {
      setNotice(e.message)
      setAskEnabled()
    }
  }

  fun isRecording(): Boolean = recorder?.isRunning == true

  private fun stopQuestionPlayback() {
    questionPlayer?.stopNow()
    questionPlayer = null
    if (recorder?.isRunning != true) runOnUiThread { setPill(null) }
  }

  private fun stopAll() {
    stopFlag = true
    stopQuestionPlayback()
    answerPlayer?.stopNow()
  }

  fun newConversation(done: ((Boolean) -> Unit)?) {
    val c = conversation
    if (busy || c == null || photo == null || !photoReady) {
      setNotice(refusal() ?: "Pick a photo first.")
      done?.invoke(false)
      return
    }
    clearTurns()
    setNotice(null)
    busy = true
    setAskEnabled()
    worker.execute {
      val ok = runCatching { c.restart() }.isSuccess
      busy = false
      setAskEnabled()
      done?.invoke(ok)
    }
  }

  // ---- the model -------------------------------------------------------------------------------------------------

  private fun afterLoad(then: () -> Unit) {
    if (engine != null) then() else {
      whenReady.add(then)
      if (!loading) loadEngine()
    }
  }

  private fun loadEngine() {
    Lfm3OmniEngine.missing(this)?.let {
      setEngineLine(it)
      failWaiting()
      return
    }
    loading = true
    val config = Lfm3OmniEngine.Config()
    worker.execute {
      try {
        val e = Lfm3OmniEngine.open(this, config) { setEngineLine(it) }
        engine = e
        conversation = Conversation(e)
        setEngineLine("Ready: ${e.placement()} (loaded in %.1f s)".format(e.load.optDouble("open_ms") / 1000))
        runOnUiThread {
          loading = false
          setAskEnabled()
          val w = whenReady
          whenReady = ArrayList()
          for (f in w) f()
          // a photo shown while an earlier load stopped at a check lost its preparation (failWaiting): prepare it now
          val p = photo
          if (w.isEmpty() && p != null && !photoReady && !busy) usePhoto(p, photoSource, example, null)
        }
      } catch (t: Throwable) {
        Log.e(Lfm3OmniEngine.TAG, "load failed", t)
        setEngineLine(if (t is Lfm3OmniEngine.NotEnoughMemory) t.message!! else "Could not load the model: ${t.message}")
        runOnUiThread {
          loading = false
          failWaiting()
        }
      }
    }
  }

  /**
   * The measuring tool: the engine opened again with another configuration (e.g. the CPU threads of each part); the
   * photo and the conversation go with the old engine. [done] gets whether the new one loaded.
   */
  fun reopenEngine(config: Lfm3OmniEngine.Config, done: (Boolean) -> Unit) {
    if (busy || loading) {
      done(false)
      return
    }
    busy = true
    photoReady = false
    setAskEnabled()
    worker.execute {
      runCatching { engine?.close() }
      engine = null
      conversation = null
      val ok = try {
        val e = Lfm3OmniEngine.open(this, config) { setEngineLine(it) }
        engine = e
        conversation = Conversation(e)
        setEngineLine("Ready: ${e.placement()} (loaded in %.1f s)".format(e.load.optDouble("open_ms") / 1000))
        true
      } catch (t: Throwable) {
        Log.e(Lfm3OmniEngine.TAG, "reload failed", t)
        setEngineLine("Could not load the model: ${t.message}")
        false
      }
      busy = false
      runOnUiThread {
        photo = null
        photoView.setImageDrawable(null)
        photoStatus.text = "No photo"
        showQuestions(null)
        clearTurns()
        setAskEnabled()
      }
      done(ok)
    }
  }

  /** The measuring tool waits for the model; without one it records the screen as it is. */
  private fun failWaiting() {
    whenReady.clear()
    autoRun?.engineUnavailable(engineLine.text.toString())
  }

  fun engineLoad(): JSONObject = engine?.load ?: JSONObject()

  /**
   * One answer. [questionEndNs] = when the question was over (the button let go, the example's last sample played, the
   * typed Ask); the first sound is timed from it. [done] gets the answer's record once its speech has played.
   */
  private fun answer(q: Conversation.Question, questionEndNs: Long, source: JSONObject, done: ((JSONObject?) -> Unit)?, opts: AskOptions) {
    refusal()?.let {
      setNotice(it)
      setAskEnabled()
      done?.invoke(null)
      return
    }
    val e = engine!!
    val c = conversation!!
    busy = true
    stopFlag = false
    setAskEnabled()
    stopButton.isEnabled = true
    val views = addTurn(q.label)
    followAnswer = true
    val runId = opts.id ?: "run%03d".format(++runCount)
    val inst = instruction
    val record = JSONObject().put("id", runId).put("question", source).put("turn", c.turns + 1).put("photo", JSONObject(photoSource.toString()))
      .put("instruction", inst).put("history", opts.history).put("seed", opts.seed).put("question_end_ns", questionEndNs)
      .put("question_end_wall_ms", System.currentTimeMillis() - (System.nanoTime() - questionEndNs) / 1_000_000)
    val ended = java.util.concurrent.CountDownLatch(1)
    val p = AudioIO.Player(Lfm3OmniEngine.SAMPLE_RATE, onSound = { ns ->
      runOnUiThread {
        setPill("● SPEAKING", SPEAKING)
        views.metrics.text = "First sound %.1f s after the question".format((ns - questionEndNs) / 1e9)
      }
    }, onEnd = { ended.countDown() })
    answerPlayer = p
    worker.execute {
      var out: Conversation.Output? = null
      try {
        out = c.ask(q, inst, opts.history, opts.sampling, opts.seed,
          replay = if (opts.fixedOrder) List(2048) { IntArray(Tokens.NUM_CODEBOOKS) { k -> k } } else null,
          listener = object : Conversation.Listener {
            override fun onAudio(pcm: FloatArray) = p.play(pcm)
            override fun onText(text: String) = runOnUiThread { views.answer.text = rendered(text) }
          }, stop = { stopFlag })
        opts.dump?.let { c.dump(File(filesDir, "dumps"), it) }
      } catch (t: Throwable) {
        Log.e(Lfm3OmniEngine.TAG, "answer failed", t)
        record.put("error", "${t.javaClass.simpleName}: ${t.message}")
        setNotice(if (t is Conversation.Full || t is UnusableAudio) t.message
        else "The answer failed: ${t.message}. The conversation starts over about the same photo.")
      }
      p.close()
      ended.await(120, java.util.concurrent.TimeUnit.SECONDS)
      // the first sound = the end of the question -> the playback head past 0 (the player's own clock reading)
      record.put("first_sound_ms", if (p.soundNs > 0) (p.soundNs - questionEndNs) / 1e6 else JSONObject.NULL)
      record.put("sound_ns", p.soundNs)
      val o = out
      if (o != null) {
        record.put("text", o.text).put("prompt_ids", Conversation.intsJson(o.promptIds.toList()))
          .put("tokens", Conversation.intsJson(o.result.tokens)).put("frames", o.result.frames.size)
          .put("timing", o.timing.put("first_codec_ms", o.timing.opt("first_audio_ns").let { ns ->
            if (ns is Long) (ns - questionEndNs) / 1e6 else JSONObject.NULL
          }).put("run_start_ms", (o.timing.getLong("run_start_ns") - questionEndNs) / 1e6))
          .put("audio_s", o.pcm.size / Lfm3OmniEngine.SAMPLE_RATE.toDouble())
          .put("player", JSONObject().put("underruns", p.underruns).put("written", p.written).put("stopped", stopFlag))
        if (opts.fixedOrder) record.put("codes", JSONArray().also { a -> for (f in o.result.frames) a.put(Conversation.intsJson(f.toList())) })
        saveRun(runId, record, o.pcm)
        runOnUiThread {
          views.answer.text = rendered(o.text)
          views.metrics.text = "%s · %.1f s of speech".format(
            record.opt("first_sound_ms").let { if (it is Number) "First sound %.1f s after the question".format(it.toDouble() / 1000) else "No speech" },
            o.pcm.size / Lfm3OmniEngine.SAMPLE_RATE.toDouble())
        }
      }
      busy = false
      runOnUiThread {
        setPill(null)
        stopButton.isEnabled = false
        setAskEnabled()
      }
      done?.invoke(record)
    }
  }

  /** files/runs/<id>.json and <id>.f32 (the answer's speech, 24 kHz float32); the last answer also as a WAV file. */
  private fun saveRun(id: String, record: JSONObject, pcm: FloatArray) {
    runCatching {
      val dir = File(filesDir, "runs").apply { mkdirs() }
      val bb = java.nio.ByteBuffer.allocate(pcm.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
      for (x in pcm) bb.putFloat(x)
      File(dir, "$id.f32").writeBytes(bb.array())
      record.put("pcm_file", "runs/$id.f32").put("pcm_samples", pcm.size)
      val part = File(dir, "$id.json.partial")
      part.writeText(record.toString(1))
      part.renameTo(File(dir, "$id.json"))
      AudioIO.writeWav(File(getExternalFilesDir(null) ?: filesDir, "last_answer.wav"), pcm, Lfm3OmniEngine.SAMPLE_RATE)
    }.onFailure { Log.w(Lfm3OmniEngine.TAG, "run record not saved: $it") }
  }

  /**
   * The model's answer as the screen shows it: **bold** in bold and a line's leading "- " or "* " as a bullet; the words
   * as the model wrote them (an unclosed ** while the answer streams is left out).
   */
  private fun rendered(text: String): CharSequence {
    val out = SpannableStringBuilder()
    for ((i, raw) in text.split('\n').withIndex()) {
      if (i > 0) out.append('\n')
      val mark = Regex("""^(\s*)[-*]\s+""").find(raw)
      val line = if (mark != null) mark.groupValues[1] + "• " + raw.substring(mark.range.last + 1) else raw
      var at = 0
      while (true) {
        val a = line.indexOf("**", at)
        val b = if (a >= 0) line.indexOf("**", a + 2) else -1
        if (a < 0 || b < 0) {
          out.append(line.substring(at).replace("**", ""))
          break
        }
        out.append(line, at, a)
        val start = out.length
        out.append(line, a + 2, b)
        out.setSpan(StyleSpan(Typeface.BOLD), start, out.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        at = b + 2
      }
    }
    return out
  }

  // ---- the measuring tool's hooks -------------------------------------------------------------------------------

  fun currentNotice(): String = notice.text.toString()
  fun currentEngineLine(): String = engineLine.text.toString()
  fun currentDeviceLine(): String = deviceLine.text.toString()
  fun currentPhotoStatus(): String = photoStatus.text.toString()
  fun currentTurns(): JSONArray = JSONArray().also { a ->
    for ((q, t) in turnTexts) a.put(JSONObject().put("question", q).put("answer", t.text.toString()))
  }
  fun exampleById(id: String): ExamplePhoto? = examples.firstOrNull { it.id == id }
  fun questionById(id: String): ExampleQuestion? = examples.flatMap { it.questions }.firstOrNull { it.id == id }

  fun keepFrames(on: Boolean) = runOnUiThread {
    heartbeat.visibility = if (on) View.VISIBLE else View.INVISIBLE
    if (on) {
      var odd = false
      val tick = object : Runnable {
        override fun run() {
          if (heartbeat.visibility != View.VISIBLE) return
          odd = !odd
          heartbeat.setBackgroundColor(if (odd) Color.rgb(254, 254, 254) else Color.WHITE)
          heartbeat.postDelayed(this, 33)
        }
      }
      heartbeat.post(tick)
    }
  }

  /** The activity's own view tree drawn into a PNG file. */
  fun snapshot(file: File) {
    runCatching {
      val v = window.decorView.rootView
      val bmp = Bitmap.createBitmap(v.width, v.height, Bitmap.Config.ARGB_8888)
      v.draw(Canvas(bmp))
      file.parentFile?.mkdirs()
      file.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }
  }

  override fun onDestroy() {
    stopFlag = true
    stopQuestionPlayback()
    answerPlayer?.stopNow()
    recorder?.stop()
    runCatching { unregisterReceiver(airplaneReceiver) }
    worker.execute { engine?.close(); engine = null }
    worker.shutdown()
    io.shutdown()
    super.onDestroy()
  }

  companion object {
    private const val PICK_PHOTO = 1
    private const val CAMERA = 2
    private const val RECORD = 3
    /** A press shorter than this is a tap: listening goes on until the next tap. */
    private const val TAP_NS = 400_000_000L
    const val DEFAULT_INSTRUCTION = ""
    private val LISTENING = Color.rgb(198, 40, 40)
    private val ASKING = Color.rgb(25, 118, 210)
    private val SPEAKING = Color.rgb(46, 125, 50)
  }
}
