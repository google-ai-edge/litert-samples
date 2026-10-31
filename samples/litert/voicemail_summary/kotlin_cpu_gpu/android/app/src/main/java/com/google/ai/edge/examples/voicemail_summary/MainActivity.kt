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

package com.google.ai.edge.examples.voicemail_summary

import android.Manifest
import android.app.Activity
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
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONObject

/**
 * One screen: pick a voicemail (an example, a recording or a file), read or edit the instruction, Summarize. The
 * answer's text appears as it is written and its speech plays as it is decoded; the number to call back gets a line
 * of its own with a Call button (the phone's dialer, nothing is dialed by the app). Every answer leaves a record in
 * files/runs/ (the timings, the text, the speech). Started with a plan (`--es plan`), the same screen is driven by
 * [AutoRun]: it fills the screen and presses the same buttons' code (the measuring tool, README.md).
 */
class MainActivity : Activity() {
  private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "litert").apply { isDaemon = true } }
  private val io = Executors.newSingleThreadExecutor { r -> Thread(r, "io").apply { isDaemon = true } }
  private var engine: Lfm3AudioEngine? = null
  private var loading = false
  private var whenReady = ArrayList<() -> Unit>()
  @Volatile private var stopFlag = false
  @Volatile private var busy = false
  private var input: Voicemail? = null
  private var inputSource: JSONObject = JSONObject()
  private var recorder: AudioIO.Recorder? = null
  private var inputPlayer: AudioIO.Player? = null
  private var answerPlayer: AudioIO.Player? = null
  private var lastPcm: FloatArray? = null
  private var runCount = 0
  private lateinit var bundled: List<BundledVoicemail>
  private var autoRun: AutoRun? = null

  private lateinit var root: LinearLayout
  private lateinit var deviceLine: TextView
  private lateinit var engineLine: TextView
  private lateinit var inputLabel: TextView
  private lateinit var playPill: TextView
  private lateinit var scroller: ScrollView
  /** While an answer comes in, the screen follows its end (and the Call back line under it); a finger on the screen stops that. */
  private var followAnswer = false
  private lateinit var playButton: Button
  private lateinit var recordButton: Button
  private lateinit var notice: TextView
  private lateinit var instruction: EditText
  private lateinit var answerIn: Spinner
  private var answerLang: AnswerLanguage = Answer.LANGUAGES.first()
  private lateinit var summarizeButton: Button
  private lateinit var stopButton: Button
  private lateinit var speakPill: TextView
  private lateinit var answer: TextView
  private lateinit var callRow: LinearLayout
  private lateinit var callLabel: TextView
  private lateinit var metrics: TextView
  private lateinit var heartbeat: View
  private var callNumber: String? = null

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
    bundled = BundledVoicemail.list(this)
    buildUi()
    val airplane = IntentFilter(Intent.ACTION_AIRPLANE_MODE_CHANGED)
    if (Build.VERSION.SDK_INT >= 33) registerReceiver(airplaneReceiver, airplane, Context.RECEIVER_NOT_EXPORTED)
    else registerReceiver(airplaneReceiver, airplane)
    selectBundled(bundled.first())
    handle(intent)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    handle(intent)
  }

  override fun onResume() {
    super.onResume()
    showDevice()
  }

  private fun handle(intent: Intent) {
    val plan = intent.getStringExtra("plan")
    when {
      plan != null -> {
        val a = AutoRun(this, plan, intent.getStringExtra("report") ?: "autorun.json", intent.getBooleanExtra("keep_frames", false))
        autoRun = a
        afterLoad { a.start() }
      }
      intent.action == Intent.ACTION_VIEW && intent.data != null -> openUri(intent.data!!)
      intent.action == Intent.ACTION_SEND -> {
        @Suppress("DEPRECATION")
        val uri = if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        else intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
        if (uri != null) openUri(uri)
      }
    }
    if (engine == null && !loading) loadEngine()
  }

  // ---- the screen ------------------------------------------------------------------------------------------------

  private fun buildUi() {
    val pad = dp(14)
    root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, pad, pad, pad) }
    scroller = ScrollView(this).apply { addView(root); setBackgroundColor(Color.WHITE) }
    scroller.setOnTouchListener { _, ev ->
      if (ev.actionMasked == MotionEvent.ACTION_DOWN) followAnswer = false
      false
    }
    root.viewTreeObserver.addOnGlobalLayoutListener { if (followAnswer) scrollToEnd() }
    ViewCompat.setOnApplyWindowInsetsListener(scroller) { v, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
      v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
      insets
    }
    setContentView(scroller)

    root.addView(text("Voicemail summary", 22f, Color.BLACK, bold = true))
    root.addView(text("An English voicemail, told back to you in your language: who called, why, and the number to call " +
      "back. LFM3-Audio-2B runs on this phone (LiteRT ${BuildConfig.LITERT_VERSION}); the app has no internet permission, so the " +
      "recording stays on the phone.", 13f, Color.DKGRAY))
    deviceLine = text("", 13f, Color.DKGRAY).apply { setPadding(0, dp(8), 0, 0) }
    root.addView(deviceLine)
    engineLine = text("", 13f, Color.DKGRAY).apply { setPadding(0, 0, 0, dp(8)) }
    root.addView(engineLine)

    root.addView(label("Voicemail"))
    val examples = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    for (v in bundled) examples.addView(Button(this).apply {
      text = v.label
      isAllCaps = false
      setOnClickListener { if (!busy) selectBundled(v) }
    })
    root.addView(HorizontalScrollView(this).apply { addView(examples) })
    val sources = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    recordButton = Button(this).apply { text = "Record"; isAllCaps = false; setOnClickListener { toggleRecord() } }
    sources.addView(recordButton)
    sources.addView(Button(this).apply { text = "Open file"; isAllCaps = false; setOnClickListener { pickFile() } })
    root.addView(sources)
    val inputRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
    playButton = Button(this).apply { text = "Play"; isAllCaps = false; setOnClickListener { togglePlayInput() } }
    inputRow.addView(playButton)
    inputLabel = text("", 15f, Color.BLACK).apply { setPadding(dp(8), 0, dp(8), 0) }
    inputRow.addView(inputLabel)
    playPill = pill("● PLAYING", Color.rgb(25, 118, 210))
    inputRow.addView(playPill)
    root.addView(inputRow)
    notice = text("", 14f, Color.rgb(183, 28, 28)).apply { visibility = View.GONE; setPadding(0, dp(4), 0, dp(4)) }
    root.addView(notice)

    val answerRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
    answerRow.addView(label("Answer in").apply { setPadding(0, 0, dp(10), 0) })
    answerIn = Spinner(this).apply {
      adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, Answer.LANGUAGES.map { it.name })
      onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
          val lang = Answer.LANGUAGES[pos]
          // a selection set by setAnswerIn (the measuring tool) keeps the instruction it was given
          if (lang == answerLang) return
          answerLang = lang
          instruction.setText(Answer.instruction(lang))
        }
        override fun onNothingSelected(p: AdapterView<*>?) {}
      }
    }
    answerRow.addView(answerIn)
    root.addView(answerRow)
    root.addView(label("Instruction"))
    instruction = EditText(this).apply {
      setText(Answer.ENGLISH_INSTRUCTION)
      minLines = 2
      gravity = Gravity.TOP
      textSize = 15f
      inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
    }
    root.addView(instruction)
    val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
    summarizeButton = Button(this).apply { text = "Summarize"; isAllCaps = false; isEnabled = false; setOnClickListener { onSummarize("tap", null) } }
    stopButton = Button(this).apply {
      text = "Stop"
      isAllCaps = false
      isEnabled = false
      setOnClickListener { stopFlag = true; answerPlayer?.stopNow() }
    }
    buttons.addView(summarizeButton)
    buttons.addView(stopButton)
    speakPill = pill("● SPEAKING", Color.rgb(46, 125, 50))
    buttons.addView(speakPill)
    root.addView(buttons)

    answer = text("", 18f, Color.BLACK).apply { setPadding(0, dp(8), 0, dp(8)); setTextIsSelectable(true) }
    root.addView(answer)
    callRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; visibility = View.GONE }
    callLabel = text("", 20f, Color.BLACK, bold = true).apply { setPadding(0, 0, dp(12), 0) }
    callRow.addView(callLabel)
    callRow.addView(Button(this).apply {
      text = "Call"
      isAllCaps = false
      setOnClickListener { callNumber?.let { startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + it.filter { c -> c.isDigit() }))) } }
    })
    root.addView(callRow)
    metrics = text("", 12f, Color.DKGRAY).apply { typeface = Typeface.MONOSPACE }
    root.addView(metrics)
    // the measuring tool's recordings: a still screen gives no video frames, so one corner pixel keeps changing
    heartbeat = View(this).apply { layoutParams = LinearLayout.LayoutParams(dp(1), dp(1)); visibility = View.INVISIBLE }
    root.addView(heartbeat)
  }

  private fun text(s: String, size: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
    text = s
    textSize = size
    setTextColor(color)
    if (bold) setTypeface(typeface, Typeface.BOLD)
  }

  private fun label(s: String) = text(s, 13f, Color.GRAY, bold = true).apply { setPadding(0, dp(10), 0, 0) }

  private fun pill(s: String, fill: Int) = text(s, 13f, Color.WHITE, bold = true).apply {
    background = GradientDrawable().apply { cornerRadius = dp(12).toFloat(); setColor(fill) }
    setPadding(dp(10), dp(3), dp(10), dp(3))
    visibility = View.INVISIBLE
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

  /** A new input clears the last answer (its text must not read as the new voicemail's). */
  private fun setInput(v: Voicemail?, source: JSONObject) = runOnUiThread {
    stopInputPlayback()
    input = v
    inputSource = source
    inputLabel.text = when {
      v == null -> "No voicemail selected"
      v.cut -> "%s · %.1f s (the first 30 s are read)".format(v.name, v.totalSeconds)
      else -> "%s · %.1f s".format(v.name, v.seconds)
    }
    playButton.isEnabled = v != null
    answer.text = ""
    metrics.text = ""
    callRow.visibility = View.GONE
    lastPcm = null
    followAnswer = false
    scroller.post { scroller.scrollTo(0, 0) }
  }

  private fun scrollToEnd() {
    val end = root.height - (scroller.height - scroller.paddingTop - scroller.paddingBottom)
    if (end > scroller.scrollY) scroller.smoothScrollTo(0, end)
  }

  fun selectBundled(b: BundledVoicemail) {
    setNotice(null)
    setInput(b.load(this), JSONObject().put("kind", "example").put("id", b.id).put("label", b.label))
  }

  // ---- inputs ----------------------------------------------------------------------------------------------------

  private fun pickFile() {
    if (busy) return
    startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
      addCategory(Intent.CATEGORY_OPENABLE)
      type = "audio/*"
    }, PICK_FILE)
  }

  @Deprecated("Activity result API of the platform")
  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    if (requestCode == PICK_FILE && resultCode == RESULT_OK) data?.data?.let { openUri(it) }
  }

  /** A picked, shared or opened file: decoded off the UI thread; [done] (the measuring tool) gets whether it is usable. */
  fun openUri(uri: Uri, done: ((Boolean) -> Unit)? = null) {
    val name = displayName(uri)
    setNotice(null)
    runOnUiThread { inputLabel.text = "Reading $name…" }
    io.execute {
      try {
        val v = AudioIO.readVoicemail(this, uri, name)
        setInput(v, JSONObject().put("kind", "file").put("label", name).put("uri", uri.toString()))
        v.shortDecode?.let { setNotice(it) }
        done?.invoke(true)
      } catch (e: UnusableAudio) {
        setInput(null, JSONObject().put("kind", "file").put("label", name).put("uri", uri.toString()))
        setNotice(e.message)
        done?.invoke(false)
      }
    }
  }

  private fun displayName(uri: Uri): String {
    if (uri.scheme == "content") runCatching {
      contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
        if (c.moveToFirst()) return c.getString(0)
      }
    }
    return uri.lastPathSegment?.substringAfterLast('/') ?: "the file"
  }

  fun toggleRecord() {
    val r = recorder
    if (r != null && r.isRunning) {
      finishRecording(r.stop())
      return
    }
    if (busy) return
    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), RECORD)
      return
    }
    startRecording()
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode != RECORD) return
    if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
      startRecording()
    } else {
      setNotice("Microphone permission was denied, so the app cannot record. Allow the microphone for Voicemail summary " +
        "in Settings (Apps > Voicemail summary > Permissions), or pick an example or a voicemail file instead.")
      autoRun?.onRecordDenied()
    }
  }

  private fun startRecording() {
    setNotice(null)
    stopInputPlayback()
    val rec = AudioIO.Recorder { s, limit ->
      runOnUiThread {
        if (limit) {
          recorder?.let { finishRecording(it.stop()) }
        } else {
          recordButton.setText("Stop recording (%d:%02d)".format(s.toInt() / 60, s.toInt() % 60))
        }
      }
    }
    try {
      rec.start()
    } catch (e: Exception) {
      setNotice("Could not start the microphone: ${e.message}")
      return
    }
    recorder = rec
    recordButton.text = "Stop recording (0:00)"
  }

  private fun finishRecording(pcm: FloatArray) {
    recorder = null
    recordButton.text = "Record"
    try {
      setInput(AudioIO.check("Recording", pcm, pcm.size / 16000.0), JSONObject().put("kind", "recording").put("label", "Recording"))
    } catch (e: UnusableAudio) {
      setInput(null, JSONObject().put("kind", "recording").put("label", "Recording"))
      setNotice(e.message)
    }
  }

  fun togglePlayInput() {
    if (inputPlayer != null) {
      stopInputPlayback()
      return
    }
    val v = input ?: return
    val p = AudioIO.Player(16000, onSound = { ns ->
      autoRun?.onInputSound(ns)
      runOnUiThread { playPill.visibility = View.VISIBLE }
    }, onEnd = { pl ->
      runOnUiThread {
        if (inputPlayer === pl) {
          inputPlayer = null
          playButton.text = "Play"
        }
        playPill.visibility = View.INVISIBLE
      }
    })
    inputPlayer = p
    playButton.text = "Stop playing"
    p.play(v.pcm16k)
    p.close()
  }

  private fun stopInputPlayback() {
    inputPlayer?.stopNow()
    inputPlayer = null
    playButton.text = "Play"
    playPill.visibility = View.INVISIBLE
  }

  // ---- the model -------------------------------------------------------------------------------------------------

  private fun afterLoad(then: () -> Unit) {
    if (engine != null) then() else {
      whenReady.add(then)
      if (!loading) loadEngine()
    }
  }

  private fun loadEngine() {
    Lfm3AudioEngine.missing(this)?.let {
      setEngineLine(it)
      failWaiting()
      return
    }
    loading = true
    val config = Lfm3AudioEngine.defaultConfig(this)
    worker.execute {
      try {
        val e = Lfm3AudioEngine.open(this, config) { setEngineLine(it) }
        engine = e
        setEngineLine("Ready: ${e.placement()} (loaded in %.1f s)".format(e.load.optDouble("open_ms") / 1000))
        runOnUiThread {
          loading = false
          summarizeButton.isEnabled = true
          val w = whenReady
          whenReady = ArrayList()
          for (f in w) f()
        }
      } catch (t: Throwable) {
        Log.e(Lfm3AudioEngine.TAG, "load failed", t)
        setEngineLine(if (t is Lfm3AudioEngine.NotEnoughMemory) t.message!! else "Could not load the model: ${t.message}")
        runOnUiThread {
          loading = false
          failWaiting()
        }
      }
    }
  }

  /** The measuring tool waits for the model; without one it records the screen as it is. */
  private fun failWaiting() {
    whenReady.clear()
    autoRun?.engineUnavailable(engineLine.text.toString())
  }

  fun engineLoad(): JSONObject = engine?.load ?: JSONObject()

  /**
   * The Summarize button (and the measuring tool, through the same code). [done] gets the answer's record once its
   * speech has played (or null when the press was refused, with the reason on the screen).
   */
  fun onSummarize(source: String, done: ((JSONObject?) -> Unit)?, seed: Long = 0, id: String? = null, pathOverride: String? = null) {
    val pressNs = System.nanoTime()
    val lang = answerLang
    val path = pathOverride ?: Answer.path(lang)
    val e = engine
    val v = input
    val text = instruction.text.toString().trim()
    val refuse = when {
      busy -> "An answer is still running."
      e == null -> engineLine.text.toString().ifEmpty { "The model is not loaded yet." }
      v == null -> "Pick an example, record a voicemail or open a file first."
      text.isEmpty() -> "Type an instruction first (the default is: ${Answer.instruction(answerLang)})"
      else -> null
    }
    if (refuse != null) {
      setNotice(refuse)
      done?.invoke(null)
      return
    }
    e!!; v!!
    stopInputPlayback()
    setNotice(null)
    busy = true
    stopFlag = false
    answer.text = ""
    metrics.text = ""
    callRow.visibility = View.GONE
    followAnswer = true
    summarizeButton.isEnabled = false
    stopButton.isEnabled = true
    val runId = id ?: "run%03d".format(++runCount)
    val record = JSONObject().put("id", runId).put("source", source).put("input", JSONObject(inputSource.toString())
      .put("seconds", v.seconds).put("total_seconds", v.totalSeconds).put("cut", v.cut))
      .put("instruction", text).put("answer_in", lang.name).put("path", path).put("seed", seed).put("press_ns", pressNs).put("press_wall_ms", System.currentTimeMillis())
    val pcmOut = java.util.Collections.synchronizedList(ArrayList<FloatArray>())
    val ended = java.util.concurrent.CountDownLatch(1)
    val p = AudioIO.Player(Lfm3AudioEngine.SAMPLE_RATE, onSound = { ns ->
      runOnUiThread {
        speakPill.visibility = View.VISIBLE
        metrics.text = "First sound %.1f s after the tap".format((ns - pressNs) / 1e9)
      }
    }, onEnd = { ended.countDown() })
    answerPlayer = p
    worker.execute {
      var out: Summarizer.Output? = null
      try {
        out = Summarizer(e).run(Summarizer.Request(v, text, lang, path, seed), object : Summarizer.Listener {
          override fun onAudio(pcm: FloatArray) {
            pcmOut.add(pcm)
            p.play(pcm)
          }
          override fun onText(text: String) = runOnUiThread { answer.text = rendered(text) }
        }, stop = { stopFlag })
      } catch (t: Throwable) {
        Log.e(Lfm3AudioEngine.TAG, "answer failed", t)
        record.put("error", "${t.javaClass.simpleName}: ${t.message}")
        setNotice("The answer failed: ${t.message}")
      }
      p.close()
      ended.await(120, java.util.concurrent.TimeUnit.SECONDS)
      // the first sound = the press -> the playback head past 0 (the player's own clock reading)
      record.put("first_sound_ms", if (p.soundNs > 0) (p.soundNs - pressNs) / 1e6 else JSONObject.NULL)
      record.put("sound_ns", p.soundNs)
      val o = out
      if (o != null) {
        lastPcm = o.pcm
        val number = Callback.findNumber(o.text)
        record.put("text", o.text).put("number_shown", number ?: JSONObject.NULL)
          .put("tokens", Summarizer.intsJson(o.result.tokens)).put("frames", o.result.frames.size)
          .put("timing", o.timing.put("first_codec_ms", o.timing.opt("first_audio_ns").let { ns ->
            if (ns is Long) (ns - pressNs) / 1e6 else JSONObject.NULL
          }).put("run_start_ms", (o.timing.getLong("run_start_ns") - pressNs) / 1e6))
          .put("audio_s", o.pcm.size / Lfm3AudioEngine.SAMPLE_RATE.toDouble())
          .put("player", JSONObject().put("underruns", p.underruns).put("written", p.written).put("stopped", stopFlag))
        bundled.firstOrNull { it.id == inputSource.optString("id") && inputSource.optString("kind") == "example" }?.let { b ->
          record.put("checks", JSONObject().put("caller", b.caller).put("callback", b.callback)
            .put("caller_named", if (lang.name == "English") Callback.namesCaller(o.text, b.caller) else Callback.namesCaller2(o.text, b.caller))
            .put("number_in_text", if (lang.name == "English") Callback.hasNumber(o.text, b.callback) else Callback.hasNumber2(o.text, b.callback)))
        }
        saveRun(runId, record, o.pcm)
        runOnUiThread {
          answer.text = rendered(o.text)
          if (number != null) {
            callNumber = number
            callLabel.text = "Call back $number"
            callRow.visibility = View.VISIBLE
          }
          metrics.text = "%s · %.1f s of speech · real-time factor %s".format(
            record.opt("first_sound_ms").let { if (it is Number) "First sound %.1f s after the tap".format(it.toDouble() / 1000) else "No speech" },
            o.pcm.size / Lfm3AudioEngine.SAMPLE_RATE.toDouble(),
            o.timing.opt("rtf").let { if (it is Number) "%.2f".format(it.toDouble()) else "-" })
        }
      }
      busy = false
      runOnUiThread {
        speakPill.visibility = View.INVISIBLE
        summarizeButton.isEnabled = true
        stopButton.isEnabled = false
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
      AudioIO.writeWav(File(getExternalFilesDir(null) ?: filesDir, "last_answer.wav"), pcm, Lfm3AudioEngine.SAMPLE_RATE)
    }.onFailure { Log.w(Lfm3AudioEngine.TAG, "run record not saved: $it") }
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
      val line = if (mark != null) mark.groupValues[1] + "\u2022 " + raw.substring(mark.range.last + 1) else raw
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

  fun isRecording(): Boolean = recorder?.isRunning == true
  fun currentNotice(): String = notice.text.toString()
  fun currentEngineLine(): String = engineLine.text.toString()
  fun currentDeviceLine(): String = deviceLine.text.toString()
  fun currentInputLabel(): String = inputLabel.text.toString()

  /** The input as the model hears it (16 kHz mono), for the measuring tool's `dump_input`. */
  fun currentInputPcm(): FloatArray? = input?.pcm16k
  fun currentAnswer(): String = answer.text.toString()
  fun setInstruction(s: String) = runOnUiThread { instruction.setText(s) }

  /** The measuring tool's "Answer in": the language and its default instruction, as a finger's choice would set them. */
  fun setAnswerIn(name: String): Boolean {
    val lang = Answer.byName(name) ?: return false
    answerLang = lang
    answerIn.setSelection(Answer.LANGUAGES.indexOf(lang))
    instruction.setText(Answer.instruction(lang))
    return true
  }
  fun bundledById(id: String): BundledVoicemail? = bundled.firstOrNull { it.id == id }

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
    stopInputPlayback()
    answerPlayer?.stopNow()
    recorder?.stop()
    runCatching { unregisterReceiver(airplaneReceiver) }
    worker.execute { engine?.close(); engine = null }
    worker.shutdown()
    io.shutdown()
    super.onDestroy()
  }

  companion object {
    private const val PICK_FILE = 1
    private const val RECORD = 2
  }
}
