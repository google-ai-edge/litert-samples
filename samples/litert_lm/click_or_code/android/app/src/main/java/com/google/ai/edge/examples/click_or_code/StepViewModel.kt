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

package com.google.ai.edge.examples.click_or_code

import android.app.Application
import android.graphics.Bitmap
import android.os.StatFs
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.examples.click_or_code.data.DownloadSafety
import com.google.ai.edge.examples.click_or_code.data.DownloadStatus
import com.google.ai.edge.examples.click_or_code.data.ModelCatalog
import com.google.ai.edge.examples.click_or_code.data.ModelStore
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/** The model file's state, from the first start to a loaded model. */
sealed class ModelState {
  data object Checking : ModelState()

  data class Missing(val bytes: Long) : ModelState()

  data class Downloading(val received: Long, val total: Long) : ModelState()

  data object Verifying : ModelState()

  data object Ready : ModelState()

  data class Error(val message: String) : ModelState()
}

/** One inference that has ended: its reasoning and the seconds of its stages. */
data class Inference(
  val thinking: String,
  val totalS: Double,
  val visionS: Double?,
  val prefillS: Double?,
  val decodeS: Double?,
  val decodeTokens: Int?,
)

/** The reply of the inference on screen, and the inferences that have ended. */
data class Decision(
  val text: String,
  val done: Boolean,
  val reply: StepReply,
  /** The screenshot's size, for the point's pixels. */
  val width: Int,
  val height: Int,
  /** 1 or 2: the inference this reply is from. */
  val inference: Int = 1,
  /** A piece of this reply has arrived. */
  val hasText: Boolean = false,
  /** From the tap on Decide to the end of the last inference. */
  val totalMs: Long? = null,
  val runs: List<Inference> = emptyList(),
  /** The reply stopped at the file's token limit, before its tool call ended. */
  val cut: Boolean = false,
  /** Stop was tapped: what is on screen is not the model's answer. */
  val stopped: Boolean = false,
  val error: String? = null,
)

data class UiState(
  val model: ModelState = ModelState.Checking,
  val engine: EngineState = EngineState.Idle,
  val photo: Bitmap? = null,
  val photoName: String? = null,
  val photoWidth: Int = 0,
  val photoHeight: Int = 0,
  val examples: List<Example> = emptyList(),
  /** The bundled requests of the screenshot on screen. */
  val requestChips: List<String> = emptyList(),
  val request: String = "",
  val deciding: Boolean = false,
  /** Stop was tapped and the step has not ended yet. */
  val stopping: Boolean = false,
  val elapsedMs: Long = 0,
  val decision: Decision? = null,
  val message: String? = null,
)

class StepViewModel(app: Application) : AndroidViewModel(app) {
  private val catalog =
    ModelCatalog.parse(app.assets.open("models.json").bufferedReader().use { it.readText() })
  private val entry = catalog.entry(ModelCatalog.COWORK)
  private val store = ModelStore(File(app.filesDir, "models"))
  private val systemPrompt =
    app.assets.open("mage_hybrid_system_prompt.txt").use { String(it.readBytes(), Charsets.UTF_8) }
  private val _ui =
    MutableStateFlow(
      UiState(
        examples =
          Examples.parse(
            app.assets.open("examples/examples.json").bufferedReader().use { it.readText() }
          )
      )
    )
  val ui: StateFlow<UiState> = _ui.asStateFlow()

  private var engine: StepEngine? = null
  private var memory = MemoryCheck()
  /** Debug builds: a file to run instead of the catalog's, for trying another conversion. */
  private var bundleOverride: File? = null
  private var photoRgb: Rgb? = null
  private var photoSha: String? = null
  private var stepJob: Job? = null
  private var downloadJob: Job? = null
  private var stateJob: Job? = null

  init {
    viewModelScope.launch { checkModel() }
  }

  private suspend fun checkModel() {
    val state = store.inspect(entry)
    // A debug launch may have named another file while the store was being read.
    if (bundleOverride != null) return
    _ui.update {
      it.copy(
        model =
          if (state.status == DownloadStatus.READY) ModelState.Ready
          else ModelState.Missing(entry.totalBytes)
      )
    }
    if (state.status == DownloadStatus.READY) startEngine()
  }

  /**
   * Side-loads a copy from the app's external files dir when one matches the catalog, else
   * downloads the model file, then loads it.
   */
  fun download() {
    if (downloadJob?.isActive == true) return
    if (bundleOverride != null) return
    downloadJob =
      viewModelScope.launch {
        try {
          val app = getApplication<Application>()
          val loaded = store.sideLoad(entry, app.getExternalFilesDir(null))
          loaded.skipped.forEach { Log.w(TAG, "side-load skipped: $it") }
          if (store.inspect(entry).status != DownloadStatus.READY) {
            val free = StatFs(app.filesDir.path).availableBytes
            val received = store.inspect(entry).receivedBytes
            // download() discards a longer partial file; a whole one goes straight to the check.
            val missing = entry.totalBytes - (if (received > entry.totalBytes) 0L else received)
            if (missing > 0 && !DownloadSafety.hasSpace(missing, free)) {
              _ui.update {
                it.copy(
                  model =
                    ModelState.Error(
                      "The download needs " +
                        "${gb(DownloadSafety.requiredFreeBytes(missing))} GB free " +
                        "and ${gb(free)} GB is free."
                    )
                )
              }
              return@launch
            }
            store.download(entry) { s ->
              _ui.update {
                it.copy(
                  model =
                    when (s.status) {
                      DownloadStatus.VERIFYING -> ModelState.Verifying
                      else -> ModelState.Downloading(s.receivedBytes, s.totalBytes)
                    }
                )
              }
            }
          }
          _ui.update { it.copy(model = ModelState.Ready) }
          startEngine()
        } catch (e: CancellationException) {
          throw e
        } catch (e: Exception) {
          Log.e(TAG, "download failed", e)
          _ui.update { it.copy(model = ModelState.Error(e.message ?: e.javaClass.simpleName)) }
        }
      }
  }

  private fun startEngine() {
    if (engine != null) return
    val app = getApplication<Application>()
    val file = bundleOverride ?: store.file(entry, entry.files.first().name)
    val step = StepEngine(file, File(app.cacheDir, "litertlm"), systemPrompt, memory)
    engine = step
    stateJob?.cancel()
    _ui.update { it.copy(engine = EngineState.Idle) }
    stateJob = viewModelScope.launch { step.state.collect { s -> _ui.update { it.copy(engine = s) } } }
    step.prepare()
  }

  /** Debug builds: run [file] instead of the catalog's model file (no hash check). */
  fun useBundle(file: File): Job =
    viewModelScope.launch {
      bundleOverride = file
      val old = engine
      engine = null
      stateJob?.cancel()
      old?.close()
      Log.i(TAG, "BUNDLE ${file.path} bytes=${file.length()}")
      _ui.update { it.copy(model = ModelState.Ready) }
      startEngine()
    }

  /** Debug builds: the memory the check asks for (the failure path of the README). */
  fun setNeededMemory(bytes: Long): Job =
    viewModelScope.launch {
      memory = MemoryCheck(bytes)
      val old = engine ?: return@launch
      engine = null
      stateJob?.cancel()
      old.close()
      startEngine()
    }

  /** Shows the picture in [bytes]; the last decision is cleared. [chips]: its bundled requests. */
  fun setPhoto(bytes: ByteArray, name: String, chips: List<String> = emptyList()): Job =
    viewModelScope.launch {
      try {
        val (rgb, screen) =
          withContext(Dispatchers.Default) { Photo.decode(bytes) to Photo.decodeForScreen(bytes) }
        photoRgb = rgb
        photoSha = sha256(bytes)
        _ui.update {
          it.copy(
            photo = screen,
            photoName = name,
            photoWidth = rgb.width,
            photoHeight = rgb.height,
            requestChips = chips,
            decision = null,
            message = null,
          )
        }
        if (BuildConfig.DEBUG)
          Log.i(TAG, "PHOTO name=$name ${rgb.width}x${rgb.height} sha256=$photoSha")
      } catch (e: CancellationException) {
        throw e
      } catch (e: Exception) {
        Log.w(TAG, "PHOTO_FAILED ${e.message}")
        _ui.update { it.copy(message = "Cannot open $name: ${e.message}") }
      }
    }

  fun setRequest(text: String) {
    _ui.update { it.copy(request = text) }
  }

  fun dismissMessage() {
    _ui.update { it.copy(message = null) }
  }

  /** Clears the request and the last decision (a take's first frame). */
  fun clearDecision() {
    _ui.update { it.copy(decision = null, request = "", message = null) }
  }

  /** Decide: the next step for the screenshot and the request on screen. */
  fun decide() {
    decideOnce(record = null)
  }

  /**
   * The next step: the agent's first step and, when its reply calls for a screenshot, its second,
   * with the same picture as the answer to that call. [record] (debug runs) gets the details. Returns
   * the job, or null when nothing was started.
   */
  fun decideOnce(
    record: JSONObject?,
    renderPrompt: Boolean = false,
    cancelAfterMs: Long? = null,
  ): Job? {
    val step = engine ?: return null
    val rgb = photoRgb ?: return null
    if (stepJob?.isActive == true) return null
    val request = _ui.value.request.trim()
    if (request.isEmpty()) {
      _ui.update { it.copy(message = "Type what you want done on this screen.") }
      record?.put("error", "no request")
      return null
    }
    val width = rgb.width
    val height = rgb.height
    val tapped = System.nanoTime()
    step.clearStop()
    val job =
      viewModelScope.launch {
        val empty = StepReply("", null, emptyList())
        _ui.update {
          it.copy(
            deciding = true,
            stopping = false,
            elapsedMs = 0,
            decision = Decision("", done = false, reply = empty, width = width, height = height),
            message = null,
          )
        }
        val ticker = launch {
          while (true) {
            delay(100)
            _ui.update { it.copy(elapsedMs = (System.nanoTime() - tapped) / 1_000_000) }
          }
        }
        val text = StringBuilder()
        val runs = mutableListOf<Inference>()
        val runRecords = org.json.JSONArray()
        val memBefore = MemoryCheck.memAvailableBytes()
        var memLow = memBefore
        var stopper: Job? = null
        try {
          val png =
            withContext(Dispatchers.Default) {
              val picture = Pictures.stretch(rgb)
              val sha = picture.sha256()
              if (BuildConfig.DEBUG) Log.i(TAG, "STEP_INPUT photo=$photoSha picture_rgb_sha256=$sha")
              record?.put("picture_rgb_sha256", sha)
              Photo.png(picture)
            }
          record?.put("prepare_ms", (System.nanoTime() - tapped) / 1_000_000)
          stopper =
            cancelAfterMs?.let { ms ->
              launch {
                delay(ms)
                stop()
              }
            }
          var firstReply: String? = null
          while (true) {
            val number = runs.size + 1
            val began = System.nanoTime()
            var firstMs: Long? = null
            var startedMs: Long? = null
            var stages: Stages? = null
            text.clear()
            _ui.update {
              it.copy(
                decision =
                  Decision("", done = false, reply = empty, width = width, height = height, inference = number, runs = runs.toList())
              )
            }
            step.step(png, request, firstReply, Step.MAX_OUTPUT_TOKENS, renderPrompt).collect { e ->
              when (e) {
                is StepEvent.Waiting -> {}
                is StepEvent.Rendered ->
                  record?.put(if (number == 1) "rendered_prompt" else "rendered_prompt_$number", e.prompt)
                is StepEvent.Started -> {
                  startedMs = (System.nanoTime() - began) / 1_000_000
                  if (number == 1) {
                    record?.put("wait_ms", e.waitMs)
                    record?.put("engine_ms", e.engineMs)
                  }
                  record?.put("conversation_ms", e.conversationMs)
                }
                is StepEvent.Text -> {
                  if (firstMs == null) firstMs = (System.nanoTime() - began) / 1_000_000
                  text.append(e.text)
                  MemoryCheck.memAvailableBytes()?.let { m ->
                    memLow = memLow?.let { minOf(it, m) } ?: m
                  }
                  _ui.update {
                    it.copy(
                      decision =
                        Decision(
                          text.toString(),
                          done = false,
                          reply = Step.parse(text.toString()),
                          width = width,
                          height = height,
                          inference = number,
                          hasText = true,
                          runs = runs.toList(),
                        )
                    )
                  }
                }
                is StepEvent.Finished -> stages = e.stages
              }
            }
            val totalS = (System.nanoTime() - began) / 1e9
            val answer = text.toString().trim()
            val reply = Step.parse(answer)
            val s = stages
            // The first piece of the reply comes after the picture's encoder and the prefill.
            val ttft = s?.timeToFirstTokenS
            val visionS =
              if (firstMs != null && startedMs != null && ttft != null) {
                ((firstMs!! - startedMs!!) / 1000.0 - ttft).coerceAtLeast(0.0)
              } else {
                null
              }
            val prefillS = s?.let { if (it.prefillTokensPerS > 0) it.prefillTokens / it.prefillTokensPerS else null }
            val decodeS = s?.let { if (it.decodeTokensPerS > 0) it.decodeTokens / it.decodeTokensPerS else null }
            runs += Inference(reply.thinking, totalS, visionS, prefillS, decodeS, s?.decodeTokens)
            runRecords.put(
              JSONObject()
                .put("inference", number)
                .put("total_s", totalS)
                .put("first_text_ms", firstMs ?: JSONObject.NULL)
                .put("vision_s", visionS ?: JSONObject.NULL)
                .put("prefill_s", prefillS ?: JSONObject.NULL)
                .put("decode_s", decodeS ?: JSONObject.NULL)
                .put("prefill_tokens", s?.prefillTokens ?: JSONObject.NULL)
                .put("decode_tokens", s?.decodeTokens ?: JSONObject.NULL)
                .put("ttft_s", ttft ?: JSONObject.NULL)
                .put("marks", JSONObject(s?.marks ?: emptyMap<String, Double>()))
                .put("text", answer)
            )
            Log.i(
              TAG,
              "INFERENCE $number total_s=$totalS first_text_ms=$firstMs vision_s=$visionS " +
                "prefill=${s?.prefillTokens} decode=${s?.decodeTokens}",
            )
            val stopped = _ui.value.stopping
            if (firstReply == null && !stopped && Step.asksForScreenshot(reply)) {
              firstReply = answer
              continue
            }
            val total = (System.nanoTime() - tapped) / 1_000_000
            val cut =
              !answer.contains("</tool_call>") &&
                (s == null || s.prefillTokens + s.decodeTokens >= Step.FILE_TOKENS - 4)
            _ui.update {
              it.copy(
                decision =
                  Decision(
                    answer,
                    done = true,
                    reply = reply,
                    width = width,
                    height = height,
                    inference = number,
                    hasText = answer.isNotEmpty(),
                    totalMs = total,
                    runs = runs.toList(),
                    cut = cut,
                    stopped = stopped,
                  )
              )
            }
            record?.put("n_inferences", number)
            record?.put("total_ms", total)
            record?.put("cut", cut)
            record?.put("stopped", stopped)
            val first = reply.first
            record?.put("route", first?.name ?: JSONObject.NULL)
            record?.put("action", first?.action ?: JSONObject.NULL)
            record?.put("strict_json", first?.strictJson ?: JSONObject.NULL)
            record?.put("action_line", reply.actionLine ?: JSONObject.NULL)
            record?.put("command", first?.command ?: JSONObject.NULL)
            first?.coordinate?.let { c ->
              val p = Step.toPixels(c, width, height)
              record?.put("point_norm", org.json.JSONArray(listOf(c.first, c.second)))
              record?.put("point_px", org.json.JSONArray(listOf(p.first, p.second)))
            }
            Log.i(
              TAG,
              "STEP inferences=$number total_ms=$total route=${first?.name} action=${first?.action} " +
                "point=${first?.coordinate} stopped=$stopped",
            )
            break
          }
        } catch (e: CancellationException) {
          throw e
        } catch (e: Exception) {
          Log.e(TAG, "STEP_FAILED ${e.javaClass.simpleName}: ${e.message}")
          record?.put("error", "${e.javaClass.simpleName}: ${e.message}")
          val shown =
            if (Step.isTooLong(e.message)) {
              "The request is too long for the model file. Shorten it and decide again."
            } else {
              e.message ?: e.javaClass.simpleName
            }
          record?.put("shown", shown)
          _ui.update {
            it.copy(
              decision =
                Decision(
                  text.toString(),
                  done = true,
                  reply = Step.parse(text.toString()),
                  width = width,
                  height = height,
                  inference = runs.size + 1,
                  runs = runs.toList(),
                  error = shown,
                )
            )
          }
        } finally {
          ticker.cancel()
          stopper?.cancel()
          withContext(NonCancellable) {
            record?.put("request", request)
            record?.put("photo_sha256", photoSha)
            record?.put("screenshot_wh", org.json.JSONArray(listOf(width, height)))
            record?.put("text", text.toString())
            record?.put("inferences", runRecords)
            record?.put("mem_available_before", memBefore ?: JSONObject.NULL)
            record?.put("mem_available_low", memLow ?: JSONObject.NULL)
            record?.put("mem_available_after", MemoryCheck.memAvailableBytes() ?: JSONObject.NULL)
            record?.put("vm_hwm_kb", vmHwmKb() ?: JSONObject.NULL)
            _ui.update { it.copy(deciding = false, stopping = false) }
          }
        }
      }
    stepJob = job
    return job
  }

  fun stop() {
    if (!_ui.value.deciding) return
    _ui.update { it.copy(stopping = true) }
    engine?.cancel()
  }

  override fun onCleared() {
    val step = engine ?: return
    engine = null
    // Not on the main thread: closing waits for a step in flight to stop (10 s at most).
    kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch { step.close() }
  }

  companion object {
    const val TAG = "ClickOrCode"

    private fun gb(bytes: Long) = "%.1f".format(bytes / 1e9)

    fun sha256(bytes: ByteArray): String =
      MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    /** The app's peak resident memory (VmHWM in /proc/self/status), in kB. */
    fun vmHwmKb(): Long? =
      runCatching {
          File("/proc/self/status").useLines { lines ->
            lines
              .firstOrNull { it.startsWith("VmHWM:") }
              ?.split(Regex("\\s+"))
              ?.getOrNull(1)
              ?.toLong()
          }
        }
        .getOrNull()
  }
}
