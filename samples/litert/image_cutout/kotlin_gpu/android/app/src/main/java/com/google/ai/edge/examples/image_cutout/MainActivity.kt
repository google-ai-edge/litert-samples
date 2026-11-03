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

package com.google.ai.edge.examples.image_cutout

import android.content.Intent
import android.content.pm.ApplicationInfo
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONObject

/** What a run ended with: a prediction, or a reason the app stopped before or after the model. */
sealed class Outcome {
  /**
   * [layers] = the screen-size cut-out and tint of the instances (made on the model thread), or
   * null.
   */
  class Done(
    val prediction: Prediction,
    val layers: Pair<android.graphics.Bitmap, android.graphics.Bitmap>?,
  ) : Outcome()

  class Failed(val kind: String, val message: String) : Outcome()
}

/**
 * One screen: a photo (Photos, Camera or Example), a task, the words, Run. The answer's boxes and
 * points are drawn on the photo, every `<OBJ>` becomes a cut-out (tap the photo to see it alone),
 * Save PNG writes the transparent PNG. Every run, from a button or from the debug autorun
 * ([AutoRun]), goes through [run].
 */
class MainActivity : ComponentActivity() {
  private lateinit var result: ResultView
  private lateinit var answerText: TextView
  private lateinit var statusText: TextView
  private lateinit var taskSpinner: Spinner
  private lateinit var formSpinner: Spinner
  private lateinit var queryEdit: EditText
  private lateinit var runButton: Button
  private lateinit var stopButton: Button
  private lateinit var saveButton: Button

  /** Every model call runs on this one thread. */
  private val worker = Executors.newSingleThreadExecutor()
  private var engine: WeVisOneEngine? = null
  private var tokenizer: QwenBpeTokenizer? = null
  private var boxFeatures: BoxFeatures? = null
  private val verified = HashMap<String, String>()

  @Volatile
  var photo: Photo? = null
    private set

  private var last: Prediction? = null
  private var exampleIndex = -1
  @Volatile private var busy = false

  /**
   * The Stop request of the current run: cleared when a run starts, set by Stop, read by the model
   * thread.
   */
  @Volatile private var stopRequested = false

  private val pickPhoto =
    registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
      if (uri != null) {
        loadPhoto { Photo.fromUri(contentResolver, uri, displayName(uri)) }
      }
    }

  private val takePhoto =
    registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
      // the camera app can take enough memory for Android to stop this process while it is open;
      // the activity is
      // then created again for the result, so the photo's path is a fixed one, not a field
      val file = cameraFile()
      if (ok && file.isFile) {
        loadPhoto { Photo.fromFile(file) }
      }
    }

  private fun cameraFile() = File(File(cacheDir, "camera"), "photo.jpg")

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    if (isDebuggable()) {
      // the autorun drives a phone nobody unlocked: stay in front (cpuset top-app) with the screen
      // on
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
      if (Build.VERSION.SDK_INT >= 27) {
        setShowWhenLocked(true)
        setTurnScreenOn(true)
      }
    }
    setContentView(R.layout.activity_main)
    // targetSdk 35 draws edge to edge: keep the controls clear of the status and navigation bars
    val root = findViewById<android.view.View>(R.id.root)
    val pad = root.paddingLeft
    androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
      val bars =
        insets.getInsets(
          androidx.core.view.WindowInsetsCompat.Type.systemBars() or
            androidx.core.view.WindowInsetsCompat.Type.ime()
        )
      v.setPadding(pad + bars.left, pad + bars.top, pad + bars.right, pad + bars.bottom)
      insets
    }
    result = findViewById(R.id.result)
    answerText = findViewById(R.id.answer)
    statusText = findViewById(R.id.status)
    taskSpinner = findViewById(R.id.task)
    formSpinner = findViewById(R.id.form)
    queryEdit = findViewById(R.id.query)
    runButton = findViewById(R.id.run)
    stopButton = findViewById(R.id.stop)
    saveButton = findViewById(R.id.save)

    taskSpinner.adapter =
      ArrayAdapter(
        this,
        android.R.layout.simple_spinner_dropdown_item,
        Task.entries.map { it.label },
      )
    taskSpinner.onItemSelectedListener =
      object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(
          parent: AdapterView<*>?,
          view: android.view.View?,
          position: Int,
          id: Long,
        ) {
          val task = Task.entries[position]
          queryEdit.hint = task.hint
          queryEdit.isEnabled = task.needsQuery
        }

        override fun onNothingSelected(parent: AdapterView<*>?) = Unit
      }
    formSpinner.adapter =
      ArrayAdapter(
        this,
        android.R.layout.simple_spinner_dropdown_item,
        FORMS.map { "LM: " + it.label },
      )
    formSpinner.setSelection(FORMS.indexOf(DEFAULT_FORM))

    findViewById<Button>(R.id.photos).setOnClickListener {
      pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
    findViewById<Button>(R.id.camera).setOnClickListener {
      val file = cameraFile().apply { parentFile?.mkdirs() }
      takePhoto.launch(FileProvider.getUriForFile(this, "$packageName.files", file))
    }
    findViewById<Button>(R.id.example).setOnClickListener { nextExample() }
    runButton.setOnClickListener {
      run(
        Task.entries[taskSpinner.selectedItemPosition],
        queryEdit.text.toString(),
        null,
        FORMS[formSpinner.selectedItemPosition],
        RunOptions(),
      ) {}
    }
    stopButton.setOnClickListener { stopRequested = true }
    saveButton.setOnClickListener { saveCutout() }
    statusText.text = "${DeviceInfo.displayName()} · LiteRT ${BuildConfig.LITERT_VERSION}"
    handleIntent(intent)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    handleIntent(intent)
  }

  private fun handleIntent(intent: Intent?) {
    val plan = intent?.getStringExtra("autorun") ?: return
    if (!isDebuggable()) return
    AutoRun(
        this,
        File(filesDir, plan),
        File(filesDir, intent.getStringExtra("out") ?: "autorun_out"),
      )
      .start()
  }

  private fun isDebuggable() = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0

  // ------------------------------------------------------------------------------------ inputs
  /**
   * The picked photo's file name from the provider (the photo picker's URI ends in a MediaStore
   * id).
   */
  private fun displayName(uri: Uri): String =
    runCatching {
      contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c
        ->
        if (c.moveToFirst()) c.getString(0) else null
      }
    }
      .getOrNull() ?: uri.lastPathSegment ?: "photo"

  private fun loadPhoto(decode: () -> Photo) {
    if (busy) return
    answerText.text = "Opening the photo"
    worker.execute {
      val p = runCatching(decode)
      runOnUiThread {
        p.onSuccess { setPhoto(it) }
          .onFailure { answerText.text = "Could not open the photo: ${it.message}" }
      }
    }
  }

  /** Shows [p] as the current photo (UI thread). */
  fun setPhoto(p: Photo?) {
    photo = p
    last = null
    saveButton.isEnabled = false
    result.setPhoto(p?.display)
    answerText.text =
      if (p == null) getString(R.string.start_hint) else "${p.name}: ${p.width} x ${p.height}"
  }

  private fun nextExample() {
    val list = examples()
    if (list.isEmpty()) return
    exampleIndex = (exampleIndex + 1) % list.size
    val e = list[exampleIndex]
    taskSpinner.setSelection(taskOf(e.getString("task")).ordinal)
    queryEdit.setText(e.getString("query"))
    val credit = e.getString("credit")
    loadPhoto {
      if (e.getString("where") == "asset") {
        Photo.fromAsset(this, "examples/" + e.getString("file"))
      } else {
        Photo.fromFile(File(File(filesDir, "examples"), e.getString("file")))
      }
    }
    statusText.text = "Example: ${e.getString("file")} (credit: $credit)"
  }

  /** The bundled examples, and the installed ones whose file is on the phone. */
  private fun examples(): List<JSONObject> {
    val doc = JSONObject(assets.open("examples/examples.json").bufferedReader().readText())
    val arr = doc.getJSONArray("examples")
    return (0 until arr.length())
      .map { arr.getJSONObject(it) }
      .filter {
        it.getString("where") == "asset" ||
          File(File(filesDir, "examples"), it.getString("file")).isFile
      }
  }

  // --------------------------------------------------------------------------------------- run
  /**
   * Runs [task] with [query] (or the exact [instruction], from the autorun) on the current photo
   * with [form]; [done] gets the outcome on the UI thread after the screen shows it.
   */
  fun run(
    task: Task,
    query: String,
    instruction: String?,
    form: DecoderForm,
    options: RunOptions,
    done: (Outcome) -> Unit,
  ) {
    if (busy) {
      done(Outcome.Failed("busy", "A run is in progress."))
      return
    }
    val p = photo
    if (p == null) {
      finish(
        Outcome.Failed(FAIL_NO_IMAGE, "Choose a photo first: Photos, Camera or Example."),
        null,
        done,
      )
      return
    }
    if (instruction == null && task.needsQuery && query.isBlank()) {
      finish(Outcome.Failed(FAIL_NO_QUERY, "Type what to look for (${task.hint})."), null, done)
      return
    }
    val text = instruction ?: Prompts.instruction(task, query)
    busy = true
    stopRequested = false
    runButton.isEnabled = false
    stopButton.isEnabled = true
    saveButton.isEnabled = false
    answerText.text = "Working"
    worker.execute {
      val outcome =
        try {
          val pr = predict(p, text, form, options)
          val layers =
            if (pr.instances.isEmpty()) {
              null
            } else {
              val t = System.nanoTime()
              displayLayers(
                  p,
                  combinedAlpha(pr.instances, p.width * p.height),
                  ResultView.COLORS[0],
                )
                .also { lastPrep["display_layers_ms"] = (System.nanoTime() - t) / 1e6 }
            }
          Outcome.Done(pr, layers)
        } catch (e: NotEnoughMemoryException) {
          Outcome.Failed(
            FAIL_MEMORY,
            "This phone does not have enough free memory for WeVisOne-VL-2B (${e.stage}): " +
              "${NotEnoughMemoryException.gb(e.availableKb)} GB available, " +
              "${NotEnoughMemoryException.gb(e.neededKb)} GB needed. " +
              "Close other apps or use a phone with " +
              "more memory; nothing was loaded.",
          )
        } catch (e: MissingFilesException) {
          Outcome.Failed(
            FAIL_FILES,
            e.message + " Install the model files with install_to_device.sh.",
          )
        } catch (e: CancelledException) {
          Outcome.Failed("stopped", "Stopped.")
        } catch (e: Throwable) {
          Log.e(TAG, "run failed", e)
          Outcome.Failed("error", "${e.javaClass.simpleName}: ${e.message}")
        }
      runOnUiThread { finish(outcome, if (outcome is Outcome.Done) task else null, done) }
    }
  }

  /**
   * Host work of the last run outside the engine's timer: the file check, the tokenizer load, the
   * screen layers.
   */
  val lastPrep = LinkedHashMap<String, Double>()

  private fun predict(
    p: Photo,
    instruction: String,
    form: DecoderForm,
    options: RunOptions,
  ): Prediction {
    lastPrep.clear()
    val dir = ModelFiles.dir(filesDir)
    val specs = ModelFiles.common() + form.file
    if (verified.isEmpty()) {
      loadVerified()
    }
    var t = System.nanoTime()
    val problems = ModelFiles.check(dir, specs, hash = true, verified = verified)
    lastPrep["file_check_ms"] = (System.nanoTime() - t) / 1e6
    saveVerified()
    if (problems.isNotEmpty()) throw MissingFilesException(problems)
    if (stopRequested) throw CancelledException()
    t = System.nanoTime()
    val tok =
      tokenizer
        ?: QwenBpeTokenizer.fromFile(File(dir, ModelFiles.TOKENIZER.name)).also {
          tokenizer = it
          lastPrep["tokenizer_load_ms"] = (System.nanoTime() - t) / 1e6
        }
    val box =
      boxFeatures
        ?: assets.open("box_log_table.bin").use { BoxFeatures.read(it) }.also { boxFeatures = it }
    var e = engine
    if (e == null || e.form != form || e.options != options) {
      e?.close()
      engine = null
      e = WeVisOneEngine(dir, cacheDir, tok, box, form, options)
      engine = e
    }
    return e.predict(p.rgb, p.width, p.height, instruction, { stopRequested }) { stage ->
      runOnUiThread { answerText.text = stage }
    }
  }

  private fun finish(outcome: Outcome, task: Task?, done: (Outcome) -> Unit) {
    busy = false
    runButton.isEnabled = true
    stopButton.isEnabled = false
    when (outcome) {
      is Outcome.Failed -> {
        answerText.text = outcome.message
        statusText.text = "${DeviceInfo.displayName()} · LiteRT ${BuildConfig.LITERT_VERSION}"
        done(outcome)
      }
      is Outcome.Done -> {
        val pr = outcome.prediction
        last = pr
        show(pr, task, outcome.layers)
        if (pr.answer.isEmpty && !pr.stopped) {
          val what = queryEdit.text.toString().trim().ifEmpty { "it" }
          val message =
            "No match for “$what” in this photo (the model answered: ${pr.answer.text})."
          answerText.text = message
          done(Outcome.Failed(FAIL_NO_TARGET, message).also { failedWith = pr })
        } else {
          done(outcome)
        }
      }
    }
  }

  /** The prediction behind a no-target outcome (for the autorun's record). */
  var failedWith: Prediction? = null
    private set

  private fun show(
    pr: Prediction,
    task: Task?,
    layers: Pair<android.graphics.Bitmap, android.graphics.Bitmap>?,
  ) {
    if (layers != null) {
      val (cut, tint) = layers
      result.setResult(pr.answer.shapes, pr.answer.segBoxes, tint, cut)
      saveButton.isEnabled = true
    } else {
      result.setResult(pr.answer.shapes, pr.answer.segBoxes, null, null)
    }
    answerText.text = summary(pr, task)
    statusText.text = statusLine(pr)
  }

  private fun summary(pr: Prediction, task: Task?): String {
    val a = pr.answer
    if (pr.stopped) return "Stopped after ${pr.newIds.size} tokens."
    val parts = ArrayList<String>()
    if (pr.instances.isNotEmpty()) {
      parts.add(
        "${pr.instances.size} cut-out${if (pr.instances.size == 1) "" else "s"} (tap the photo)"
      )
    }
    val byLabel = a.shapes.groupBy { it.label }
    for ((label, shapes) in byLabel) {
      parts.add(if (shapes.size == 1) label else "$label × ${shapes.size}")
    }
    a.statedCount?.let { parts.add("total: $it") }
    if (task == Task.READ_TEXT) return "Text: " + a.labels().joinToString(" ")
    return parts.joinToString(" · ").ifEmpty { a.text }
  }

  private fun statusLine(pr: Prediction): String {
    val t = pr.timing
    fun s(key: String) = (t[key] as? Double)?.let { "%.1f s".format(it / 1000) } ?: "-"
    val mask = (t["mask_decoder_ms"] as? List<*>)?.sumOf { it as Double }
    val form = engine?.form?.label ?: ""
    return "${DeviceInfo.displayName()} · LM $form · " +
      "${s("total_ms")}: vision ${s("vision_encoder_ms")}, " +
      "prefill ${s("prefill_ms")}, ${pr.newIds.size} tokens ${s("decode_ms")}" +
      (mask?.let { ", cut-out %.1f s".format(it / 1000) } ?: "")
  }

  private fun saveCutout() {
    val p = photo ?: return
    val pr = last ?: return
    if (pr.instances.isEmpty()) return
    saveButton.isEnabled = false
    worker.execute {
      val message = runCatching {
        val bitmap = cutoutBitmap(p, combinedAlpha(pr.instances, p.width * p.height))
        val where = savePng(this, bitmap, "wevisone_${System.currentTimeMillis()}.png")
        bitmap.recycle()
        "Saved $where"
      }
        .getOrElse { "Could not save: ${it.message}" }
      runOnUiThread {
        statusText.text = message
        saveButton.isEnabled = true
      }
    }
  }

  /** Runs [block] on the model thread (the autorun's file writes) and waits for it. */
  fun onWorker(block: () -> Unit) {
    val latch = java.util.concurrent.CountDownLatch(1)
    worker.execute {
      try {
        block()
      } finally {
        latch.countDown()
      }
    }
    latch.await()
  }

  fun setInputs(task: Task, query: String, form: DecoderForm? = null) {
    taskSpinner.setSelection(task.ordinal)
    queryEdit.setText(query)
    if (form != null) {
      formSpinner.setSelection(FORMS.indexOf(form).coerceAtLeast(0))
    }
  }

  fun currentEngineForm(): DecoderForm? = engine?.form

  private fun loadVerified() {
    val prefs = getSharedPreferences("verified", MODE_PRIVATE)
    for ((k, v) in prefs.all) {
      if (v is String) {
        verified[k] = v
      }
    }
  }

  private fun saveVerified() {
    val edit = getSharedPreferences("verified", MODE_PRIVATE).edit().clear()
    for ((k, v) in verified) {
      edit.putString(k, v)
    }
    edit.apply()
  }

  override fun onDestroy() {
    worker.execute { engine?.close() }
    worker.shutdown()
    super.onDestroy()
  }

  companion object {
    const val TAG = "WeVisOne"
    const val FAIL_NO_IMAGE = "no_image"
    const val FAIL_NO_QUERY = "no_query"
    const val FAIL_NO_TARGET = "no_target"
    const val FAIL_MEMORY = "memory"
    const val FAIL_FILES = "files"

    /**
     * The language model forms offered on the screen. The default is the fastest form that matched
     * the authors' fp32 model on the Galaxy S26 (README, "How the answers were checked"): int8
     * weights on the GPU.
     */
    val FORMS =
      listOf(
        DecoderForm.GPU_INT8_FP32,
        DecoderForm.GPU_FP16_ACC32,
        DecoderForm.CPU_FP32,
        DecoderForm.GPU_FP32,
      )
    val DEFAULT_FORM = DecoderForm.GPU_INT8_FP32
  }
}
