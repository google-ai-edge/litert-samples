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

package com.google.ai.edge.examples.offline_planner

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.net.ConnectivityManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.provider.Settings
import android.text.InputType
import android.util.Log
import android.util.TypedValue
import android.view.Choreographer
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import java.io.File
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.LocalTime
import java.time.ZoneId
import org.json.JSONArray
import org.json.JSONObject

/**
 * The screen: a request (typed or picked from the examples), Ask, the actions the model proposes
 * with a check on each, Run, and the app's own calendar, outbox and contacts, which the user can
 * edit. The measuring tool ([AutoRun]) drives the same functions the buttons call.
 */
class MainActivity : Activity() {
  private val main = Handler(Looper.getMainLooper())
  private lateinit var stateFile: File
  private lateinit var state: PlannerState
  private var engine: PlannerEngine? = null
  private var modelFile: File? = null

  /** The clock the prompt and the example data use: the phone's, or a plan's fixed time. */
  private var fixedNow: LocalDateTime? = null

  private lateinit var statusView: TextView
  private lateinit var networkView: TextView
  private lateinit var requestView: EditText
  private lateinit var askButton: Button
  private lateinit var runButton: Button
  private lateinit var rawButton: Button
  private lateinit var timingView: TextView
  private lateinit var actionsBox: LinearLayout
  private lateinit var answerView: TextView
  private lateinit var rawView: TextView
  private lateinit var doneBox: LinearLayout
  private lateinit var calendarBox: LinearLayout
  private lateinit var outboxBox: LinearLayout
  private lateinit var contactsBox: LinearLayout
  private lateinit var scroll: ScrollView
  private lateinit var doneHeader: TextView

  /** Why the model could not be opened or loaded, if it could not. */
  @Volatile var loadError: String? = null
    private set

  private var actions: List<Action> = emptyList()
  private var changedEvents = setOf<Long>()
  private var changedDrafts = setOf<Long>()
  private var busy = false

  /** The request in flight or the last one: what the measuring tool reads. */
  @Volatile var last: RequestRecord? = null
  private var autoRun: AutoRun? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    stateFile = File(filesDir, "planner_state.json")
    state = PlannerState.load(stateFile) ?: PlannerState.seed(now()).also { it.save(stateFile) }
    buildScreen()
    refreshLists()
    if (!handleExtras(intent)) openModel(DEFAULT_MODEL, useGpu = true)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    handleExtras(intent)
  }

  override fun onResume() {
    super.onResume()
    showNetwork()
  }

  override fun onDestroy() {
    autoRun?.stop()
    engine?.close()
    super.onDestroy()
  }

  fun now(): LocalDateTime = fixedNow ?: LocalDateTime.now().withSecond(0).withNano(0)

  // ---- Model ------------------------------------------------------------------------------

  /** Opens [fileName] from the app's external files dir and prepares the first Engine. */
  fun openModel(fileName: String, useGpu: Boolean) {
    engine?.close()
    engine = null
    val dir = getExternalFilesDir(null)
    val file = File(dir, fileName)
    modelFile = file
    loadError = null
    if (!file.isFile) {
      loadError = "model file not found"
      setStatus("Model file not found: ${file.absolutePath}\nCopy it to the phone with install_to_device.sh (see the README).")
      askButton.isEnabled = false
      return
    }
    val e = PlannerEngine(file.absolutePath, cacheDir.absolutePath, useGpu)
    engine = e
    setStatus("Loading ${file.name} on the ${e.backendName}…")
    askButton.isEnabled = false
    prepareEngine(e)
  }

  private fun prepareEngine(e: PlannerEngine) {
    e.prepare { result ->
      main.post {
        if (engine !== e) return@post
        result
          .onSuccess { ms ->
            setStatus("Ready · ${e.backendName} · model loaded in ${"%.1f".format(ms / 1000.0)} s")
            askButton.isEnabled = !busy
          }
          .onFailure {
            loadError = "${it.javaClass.simpleName}: ${it.message}"
            setStatus("Could not load the model on the ${e.backendName}: ${it.message}")
          }
      }
    }
  }

  // ---- Ask and Run ------------------------------------------------------------------------

  /** Ask: sends the request to the model. Returns false when nothing was sent. */
  fun ask(): Boolean {
    val request = requestView.text.toString().trim()
    val e = engine
    if (request.isEmpty()) {
      clearResult()
      answerView.text = "Type a request, or pick one of the examples."
      answerView.visibility = View.VISIBLE
      last = RequestRecord(request, SystemClock.elapsedRealtimeNanos(), -1).also { it.finish(problem = "empty request") }
      return false
    }
    if (e == null || !e.ready || busy) return false
    val tap = SystemClock.elapsedRealtimeNanos()
    getSystemService(InputMethodManager::class.java)?.hideSoftInputFromWindow(requestView.windowToken, 0)
    busy = true
    askButton.isEnabled = false
    clearResult()
    val record = RequestRecord(request, tap, e.lastInitMs)
    last = record
    setStatus("Working… (${e.backendName})")
    val zone = ZoneId.systemDefault()
    val system = Planner.systemInstruction(now(), zone, state)
    record.system = system
    e.ask(
      system,
      request,
      object : PlannerEngine.Listener {
        override fun onText(text: String) {
          main.post { showPartial(record, text) }
        }

        override fun onDone(reply: PlannerEngine.Reply) {
          main.post { showReply(record, reply) }
        }

        override fun onError(message: String) {
          main.post {
            busy = false
            setStatus(message)
            // The status line changes when the next Engine is ready; the answer area keeps the reason.
            answerView.text = message
            answerView.visibility = View.VISIBLE
            record.finish(problem = message)
            saveRun(record)
            prepareEngine(e)
          }
        }
      },
    )
    return true
  }

  /** The reply is still streaming: show each call as soon as it is complete. */
  private fun showPartial(record: RequestRecord, text: String) {
    if (record !== last || record.done) return
    val parsed = ReplyParser.parse(text, complete = false)
    if (parsed.calls.size > actions.size) {
      actions = Actions.check(parsed.calls, state)
      renderActions()
      if (record.firstCallNs == 0L) markNextFrame { record.firstCallNs = it }
    }
    timingView.text = "Writing… ${"%.1f".format((SystemClock.elapsedRealtimeNanos() - record.tapNs) / 1e9)} s"
  }

  private fun showReply(record: RequestRecord, reply: PlannerEngine.Reply, fromModel: Boolean = true) {
    busy = false
    val parsed = ReplyParser.parse(reply.text, complete = true)
    actions = Actions.check(parsed.calls, state)
    renderActions()
    rawView.text = reply.text + if (reply.thought.isBlank()) "" else "\n\n[thought channel]\n${reply.thought}"
    if (parsed.calls.isEmpty() || parsed.problem != null) {
      answerView.text =
        when {
          parsed.problem != null -> "Could not read the reply as tool calls: ${parsed.problem} The model's text is below."
          parsed.answer.isNotEmpty() -> parsed.answer
          else -> "The model returned no text."
        }
      answerView.visibility = View.VISIBLE
      if (parsed.problem != null) rawView.visibility = View.VISIBLE
    }
    runButton.isEnabled = actions.any { it.ok }
    record.reply = reply
    record.parsed = parsed
    record.actions = actions
    markNextFrame { shown ->
      record.shownNs = shown
      record.finish(problem = parsed.problem)
      saveRun(record)
      val secs = (shown - record.tapNs) / 1e9
      val first = if (record.firstCallNs > 0) " · first call ${"%.2f".format((record.firstCallNs - record.tapNs) / 1e9)} s" else ""
      timingView.text = "Reply in ${"%.2f".format(secs)} s$first · ${engine?.backendName ?: ""}"
    }
    // A fresh Engine for the next request (the model's state must not carry over).
    if (fromModel) {
      engine?.let {
        setStatus("Preparing a fresh engine for the next request…")
        prepareEngine(it)
      }
    }
  }

  /** Run: carries out the checked actions, in order, on the app's own data. */
  fun runActions(): List<String> {
    val lines = mutableListOf<String>()
    val events = mutableSetOf<Long>()
    val drafts = mutableSetOf<Long>()
    for (a in actions) {
      if (!a.ok) {
        lines += "✗ Skipped: ${a.summary} (${a.refusal})"
        continue
      }
      val c = a.run()
      lines += "✓ ${c.line}"
      c.eventId?.let { events += it }
      c.draftId?.let { drafts += it }
    }
    state.save(stateFile)
    changedEvents = events
    changedDrafts = drafts
    actions = emptyList()
    renderActions()
    runButton.isEnabled = false
    doneBox.removeAllViews()
    lines.forEach { doneBox.addView(text(it, 15f, if (it.startsWith("✓")) GREEN else RED)) }
    doneHeader.visibility = View.VISIBLE
    refreshLists()
    last?.let {
      it.runLines = lines
      saveRun(it)
    }
    scroll.post { scroll.smoothScrollTo(0, doneHeader.top - dp(8)) }
    return lines
  }

  private fun clearResult() {
    actions = emptyList()
    renderActions()
    answerView.visibility = View.GONE
    rawView.visibility = View.GONE
    rawView.text = ""
    timingView.text = ""
    runButton.isEnabled = false
    doneBox.removeAllViews()
    doneHeader.visibility = View.GONE
  }

  private var runCount = 0
  private var runFile: File? = null

  /** Every request leaves its record in `runs/` of the app's internal files dir (the request,
   * the reply and the times). Run rewrites it. */
  private fun saveRun(record: RequestRecord) {
    val dir = File(filesDir, "runs").apply { mkdirs() }
    val file =
      if (record.runLines != null && runFile != null) runFile!!
      else File(dir, "run_%s_%03d.json".format(java.time.LocalDateTime.now().format(RUN_NAME), ++runCount))
    runFile = file
    runCatching { file.writeText(record.toJson().put("backend", engine?.backendName ?: "").toString(1)) }
  }

  /** Calls [block] with the time of the next frame, the one that shows the views just changed. */
  private fun markNextFrame(block: (Long) -> Unit) {
    Choreographer.getInstance().postFrameCallback { block(SystemClock.elapsedRealtimeNanos()) }
  }

  // ---- The screen -------------------------------------------------------------------------

  private fun buildScreen() {
    val root = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(dp(16), dp(8), dp(16), dp(24))
    }
    scroll = ScrollView(this).apply {
      addView(root)
      isFillViewport = true
    }
    // The system bars and the keyboard pad the frame around the scroll view, so the scroll view gets shorter when the
    // keyboard opens and scrolls the focused field above it (with the padding on the scroll view itself, a field near
    // the bottom stayed under the keyboard).
    val frame = FrameLayout(this).apply {
      addView(scroll)
      setBackgroundColor(Color.WHITE)
    }
    frame.setOnApplyWindowInsetsListener { v, insets ->
      val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.ime())
      v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
      WindowInsets.CONSUMED
    }
    setContentView(frame)
    // Dark status and navigation bar icons on the white screen.
    val light = WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
    window.insetsController?.setSystemBarsAppearance(light, light)

    networkView = text("", 13f, BLUE).apply { setTypeface(typeface, Typeface.BOLD) }
    statusView = text("", 13f, GREY)
    root.addView(networkView)
    root.addView(statusView)
    root.addView(text("Offline Planner", 24f, Color.BLACK).apply {
      setTypeface(typeface, Typeface.BOLD)
      setPadding(0, dp(8), 0, 0)
    })
    root.addView(text("Ask in one sentence. LFM3-1.7B on the phone proposes calendar and message actions; nothing changes until you tap Run.", 14f, GREY))

    requestView = EditText(this).apply {
      hint = "e.g. ${Planner.EXAMPLES[1]}"
      inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
      minLines = 2
      maxLines = 4
      setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
    }
    root.addView(requestView, matchWidth().apply { topMargin = dp(8) })

    val examples = Spinner(this)
    examples.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, listOf("Examples") + Planner.EXAMPLES)
    examples.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
      override fun onItemSelected(p: AdapterView<*>?, v: View?, position: Int, id: Long) {
        if (position > 0) {
          requestView.setText(Planner.EXAMPLES[position - 1])
          examples.setSelection(0)
        }
      }

      override fun onNothingSelected(p: AdapterView<*>?) {}
    }
    askButton = Button(this).apply {
      text = "Ask"
      isEnabled = false
      setOnClickListener { ask() }
    }
    root.addView(row(examples to 1f, askButton to 0f))

    timingView = text("", 13f, GREY)
    root.addView(header("Proposed actions"))
    root.addView(timingView)
    actionsBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    root.addView(actionsBox)
    answerView = text("", 16f, Color.BLACK).apply { visibility = View.GONE }
    root.addView(answerView)
    runButton = Button(this).apply {
      text = "Run"
      isEnabled = false
      setOnClickListener { runActions() }
    }
    rawButton = Button(this).apply {
      text = "Model output"
      setOnClickListener { rawView.visibility = if (rawView.visibility == View.VISIBLE) View.GONE else View.VISIBLE }
    }
    root.addView(row(runButton to 1f, rawButton to 0f))
    rawView = text("", 12f, Color.DKGRAY).apply {
      typeface = Typeface.MONOSPACE
      visibility = View.GONE
      setBackgroundColor(0xFFF2F2F2.toInt())
      setPadding(dp(8), dp(8), dp(8), dp(8))
      setTextIsSelectable(true)
    }
    root.addView(rawView)
    doneHeader = header("Done").apply { visibility = View.GONE }
    root.addView(doneHeader)
    doneBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    root.addView(doneBox)

    root.addView(header("Calendar"))
    calendarBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    root.addView(calendarBox)
    root.addView(addEventRow())
    root.addView(header("Outbox (drafts, not sent)"))
    outboxBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    root.addView(outboxBox)
    root.addView(header("Contacts"))
    contactsBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    root.addView(contactsBox)
    root.addView(addContactRow())
    root.addView(Button(this).apply {
      text = "Reset the example data"
      setOnClickListener { resetState(now()) }
    }, matchWidth().apply { topMargin = dp(16) })
  }

  private fun renderActions() {
    actionsBox.removeAllViews()
    actions.forEachIndexed { i, a ->
      val box = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(10), dp(8), dp(10), dp(8))
        setBackgroundColor(if (a.ok) 0xFFF1F6FE.toInt() else 0xFFFDECEC.toInt())
      }
      box.addView(text("${i + 1}. ${a.summary}", 16f, Color.BLACK))
      box.addView(text(if (a.ok) "${a.call.name} · ready to run" else "${a.call.name} · will not run: ${a.refusal}", 13f, if (a.ok) GREY else RED))
      actionsBox.addView(box, matchWidth().apply { topMargin = dp(6) })
    }
  }

  fun refreshLists() {
    calendarBox.removeAllViews()
    for (e in state.sortedEvents()) {
      val mark = if (e.id in changedEvents) "  ● changed" else ""
      calendarBox.addView(itemRow("${Actions.show(e.start)}  ${e.title}$mark", e.id in changedEvents) {
        state.events.removeAll { it.id == e.id }
        state.save(stateFile)
        refreshLists()
      })
    }
    outboxBox.removeAllViews()
    if (state.drafts.isEmpty()) outboxBox.addView(text("No drafts.", 14f, GREY))
    for (d in state.drafts) {
      outboxBox.addView(itemRow("To ${d.recipient}: \"${d.message}\"", d.id in changedDrafts) {
        state.drafts.removeAll { it.id == d.id }
        state.save(stateFile)
        refreshLists()
      })
    }
    contactsBox.removeAllViews()
    for (c in state.contacts) {
      contactsBox.addView(itemRow(c.name, false) {
        state.contacts.removeAll { it.name == c.name }
        state.save(stateFile)
        refreshLists()
      })
    }
  }

  fun resetState(at: LocalDateTime) {
    state = PlannerState.seed(at)
    state.save(stateFile)
    changedEvents = emptySet()
    changedDrafts = emptySet()
    clearResult()
    refreshLists()
    scroll.scrollTo(0, 0)
  }

  private fun addEventRow(): View {
    val title = EditText(this).apply { hint = "Title"; setSingleLine() }
    val date = EditText(this).apply { hint = "YYYY-MM-DD"; setSingleLine() }
    val time = EditText(this).apply { hint = "HH:MM"; setSingleLine() }
    val add = Button(this).apply {
      text = "Add"
      setOnClickListener {
        val d = runCatching { LocalDate.parse(plain(date.text.toString())) }.getOrNull()
        val t = runCatching { LocalTime.parse(plain(time.text.toString()).padStart(5, '0')) }.getOrNull()
        if (title.text.isBlank() || d == null || t == null) {
          setStatus("To add an event, fill in a title, a date (YYYY-MM-DD) and a time (HH:MM).")
        } else {
          state.addEvent(title.text.toString(), d.atTime(t))
          state.save(stateFile)
          title.setText(""); date.setText(""); time.setText("")
          refreshLists()
        }
      }
    }
    return row(title to 1.4f, date to 1.2f, time to 0.8f, add to 0f)
  }

  /** A typed date or time as plain ASCII: a Japanese keyboard, for one, types full-width digits and a minus sign. */
  private fun plain(s: String): String =
    java.text.Normalizer.normalize(s, java.text.Normalizer.Form.NFKC).replace(Regex("[\u2010-\u2015\u2212\u30fc]"), "-").trim()

  private fun addContactRow(): View {
    val name = EditText(this).apply { hint = "New contact's name"; setSingleLine() }
    val add = Button(this).apply {
      text = "Add"
      setOnClickListener {
        if (state.addContact(name.text.toString())) {
          state.save(stateFile)
          name.setText("")
          refreshLists()
        }
      }
    }
    return row(name to 1f, add to 0f)
  }

  private fun setStatus(s: String) {
    statusView.text = s
    showNetwork()
  }

  /** The device, the runtime and whether the phone is online. */
  private fun showNetwork() {
    if (!::networkView.isInitialized) return
    val airplane = Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
    val online = getSystemService(ConnectivityManager::class.java)?.activeNetwork != null
    val net = (if (airplane) "Airplane mode on" else "Airplane mode off") + if (online) " · online" else " · no network"
    networkView.text = "$net · ${deviceName()} · LiteRT-LM $LITERT_LM_VERSION"
    networkView.setTextColor(if (online) GREY else BLUE)
  }

  /** The phone's model (not the user-set device name, which can hold a person's name). */
  fun deviceName(): String = Build.MODEL

  fun networkState(): Pair<Boolean, Boolean> {
    val airplane = Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
    val online = getSystemService(ConnectivityManager::class.java)?.activeNetwork != null
    return airplane to online
  }

  fun thermalStatus(): Int = getSystemService(PowerManager::class.java)?.currentThermalStatus ?: -1

  fun engineReady(): Boolean = engine?.ready == true && !busy

  fun backendName(): String = engine?.backendName ?: ""

  fun modelFile(): File? = modelFile

  fun setRequest(s: String) {
    requestView.setText(s)
    requestView.setSelection(requestView.text.length)
  }

  /** Puts [reply] through the same parse, check and display as a model reply (the measuring
   * tool's failure-path items). */
  fun showInjectedReply(reply: String) {
    clearResult()
    val record = RequestRecord(requestView.text.toString(), SystemClock.elapsedRealtimeNanos(), -1)
    record.injected = true
    last = record
    showReply(record, PlannerEngine.Reply(reply, "", 0, SystemClock.elapsedRealtimeNanos(), -1, 0), fromModel = false)
  }

  /** The screen's lines, for the measuring tool's report. */
  fun screenLines(): Map<String, Any> =
    mapOf(
      "network" to networkView.text.toString(),
      "status" to statusView.text.toString(),
      "timing" to timingView.text.toString(),
      "answer" to if (answerView.visibility == View.VISIBLE) answerView.text.toString() else "",
      "actions" to actions.map { if (it.ok) "${it.summary} [ready]" else "${it.summary} [will not run: ${it.refusal}]" },
      "run_enabled" to runButton.isEnabled,
      "ask_enabled" to askButton.isEnabled,
    )

  fun stateJson() = state.toJson()

  fun setFixedNow(t: LocalDateTime?) {
    fixedNow = t
  }

  // ---- Measuring tool -------------------------------------------------------------------

  /** Starts a plan from the launch extras (debug builds). Returns true when it opened a model. */
  private fun handleExtras(intent: Intent): Boolean {
    if (!BuildConfig.DEBUG) return false
    val plan = intent.getStringExtra("plan") ?: return false
    // Over the keyguard and awake while a plan runs: a phone driven from adb is often locked.
    setShowWhenLocked(true)
    setTurnScreenOn(true)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    val model = intent.getStringExtra("model")
    val backend = intent.getStringExtra("backend")
    val opens = model != null || backend != null
    if (opens) openModel(model ?: DEFAULT_MODEL, useGpu = backend?.lowercase() != "cpu")
    autoRun?.stop()
    autoRun = AutoRun(this, plan, intent.getStringExtra("report") ?: "report").also { it.start() }
    return opens
  }

  // ---- View helpers -----------------------------------------------------------------------

  private fun text(s: String, sp: Float, color: Int) =
    TextView(this).apply {
      text = s
      setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
      setTextColor(color)
    }

  private fun header(s: String) =
    text(s, 17f, Color.BLACK).apply {
      setTypeface(typeface, Typeface.BOLD)
      setPadding(0, dp(14), 0, dp(2))
    }

  private fun itemRow(s: String, highlight: Boolean, onRemove: () -> Unit): View {
    val label = text(s, 15f, Color.BLACK)
    val remove = TextView(this).apply {
      text = "✕"
      setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
      setTextColor(GREY)
      setPadding(dp(12), dp(6), dp(4), dp(6))
      contentDescription = "Remove"
      setOnClickListener { onRemove() }
    }
    return row(label to 1f, remove to 0f).apply {
      setPadding(dp(6), dp(4), dp(6), dp(4))
      if (highlight) setBackgroundColor(0xFFE3F4E6.toInt())
    }
  }

  private fun row(vararg views: Pair<View, Float>) =
    LinearLayout(this).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
      for ((v, w) in views) {
        addView(v, if (w > 0f) LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, w)
          else LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT))
      }
    }

  private fun matchWidth() =
    LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

  private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

  companion object {
    const val DEFAULT_MODEL = "LFM3-1.7B_mixed_int4_ekv2048.litertlm"
    private val RUN_NAME = java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")
    const val LITERT_LM_VERSION = "0.18.1"
    private val GREY = 0xFF5F6368.toInt()
    private val BLUE = 0xFF1A73E8.toInt()
    private val GREEN = 0xFF188038.toInt()
    private val RED = 0xFFC5221F.toInt()
  }
}

/** One request as the measuring tool records it. Times are `elapsedRealtimeNanos()`. */
class RequestRecord(val request: String, val tapNs: Long, val engineInitMs: Long) {
  var system: String? = null
  var reply: PlannerEngine.Reply? = null
  var parsed: ParsedReply? = null
  var actions: List<Action> = emptyList()
  var firstCallNs = 0L
  var shownNs = 0L
  var injected = false
  var runLines: List<String>? = null
  var problem: String? = null

  @Volatile var done = false

  fun finish(problem: String?) {
    this.problem = problem
    done = true
    Log.i("OfflinePlanner", "request done: ${request.take(60)} problem=$problem")
  }

  /** The record as JSON: the reply, the calls, the checks and the times in ms from the tap. */
  fun toJson(): JSONObject {
    val ms = { t: Long -> if (t > 0) (t - tapNs) / 1e6 else JSONObject.NULL }
    return JSONObject()
      .put("request", request)
      .put("injected", injected)
      .put("system", system ?: JSONObject.NULL)
      .put("raw", reply?.text ?: JSONObject.NULL)
      .put("thought", reply?.thought ?: JSONObject.NULL)
      .put("calls", JSONArray(parsed?.calls.orEmpty().map { JSONObject().put("name", it.name).put("args", JSONObject(it.args)) }))
      .put("actions", JSONArray(actions.map { JSONObject().put("summary", it.summary).put("ok", it.ok).put("refusal", it.refusal ?: JSONObject.NULL) }))
      .put("answer", parsed?.answer ?: JSONObject.NULL)
      .put("problem", problem ?: JSONObject.NULL)
      .put("first_token_ms", ms(reply?.firstTokenNs ?: 0L))
      .put("first_call_ms", ms(firstCallNs))
      .put("reply_done_ms", ms(reply?.doneNs ?: 0L))
      .put("shown_ms", ms(shownNs))
      .put("kv_tokens", reply?.kvTokens ?: -1)
      .put("chunks", reply?.chunks ?: 0)
      .put("engine_init_ms", engineInitMs)
      .put("run", runLines?.let { JSONArray(it) } ?: JSONObject.NULL)
  }
}
