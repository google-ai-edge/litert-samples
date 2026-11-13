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

package com.google.ai.edge.examples.voice_to_text

import android.app.Application
import android.os.StatFs
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.examples.voice_to_text.data.DownloadSafety
import com.google.ai.edge.examples.voice_to_text.data.DownloadStatus
import com.google.ai.edge.examples.voice_to_text.data.ModelCatalog
import com.google.ai.edge.examples.voice_to_text.data.ModelStore
import java.io.File
import kotlin.math.abs
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

/** The model file's state, from the first start to a file on the phone. */
sealed class ModelState {
  data object Checking : ModelState()

  data class Missing(val bytes: Long) : ModelState()

  /** Part of the file is on the phone; Download resumes from there. */
  data class Paused(val received: Long, val total: Long) : ModelState()

  data class Downloading(val received: Long, val total: Long) : ModelState()

  data object Verifying : ModelState()

  data object Ready : ModelState()

  data class Error(val message: String) : ModelState()
}

/** What the message line offers besides OK. */
enum class MessageAction {
  OPEN_SETTINGS,
}

/** One recording's text, as the screen shows it. */
data class Transcript(
  val id: Int,
  val task: Task,
  val keywords: List<String>,
  val seconds: Double,
  val text: String,
  val done: Boolean,
  val cancelled: Boolean = false,
  /** From the end of the recording (Stop) to the first piece of text, and to the last. */
  val firstTextMs: Long? = null,
  val totalMs: Long? = null,
  val error: String? = null,
)

data class UiState(
  val model: ModelState = ModelState.Checking,
  val engine: EngineState = EngineState.Idle,
  val task: Task = Task.TRANSCRIPT,
  val keywords: String = "",
  val recording: Boolean = false,
  val recordedMs: Long = 0,
  val maxSeconds: Int = Prompts.MAX_SECONDS,
  /** The last 20 ms's peak, 0..1, for the level bar. */
  val level: Float = 0f,
  val transcribing: Boolean = false,
  /** Stop was tapped during a transcription and it has not ended yet. */
  val stopping: Boolean = false,
  val hasLastRecording: Boolean = false,
  /** Newest first. */
  val results: List<Transcript> = emptyList(),
  val message: String? = null,
  val messageAction: MessageAction? = null,
)

class VoiceViewModel(app: Application) : AndroidViewModel(app) {
  private val catalog =
    ModelCatalog.parse(app.assets.open("models.json").bufferedReader().use { it.readText() })
  private val entry = catalog.entry(ModelCatalog.SPEECH)
  private val store = ModelStore(File(app.filesDir, "models"))
  private val recorder = Recorder()
  private val _ui = MutableStateFlow(UiState())
  val ui: StateFlow<UiState> = _ui.asStateFlow()

  private var engine: SpeechEngine? = null
  private var memory = MemoryCheck()
  private var lastRecording: FloatArray? = null
  private var recordJob: Job? = null
  private var transcribeJob: Job? = null
  private var downloadJob: Job? = null
  private var stateJob: Job? = null
  private var nextId = 1

  /** The transcription in flight has sent its message (Stop then goes to the runtime). */
  @Volatile private var started = false

  /** Debug builds: where each recording and its text are written (null = nowhere). */
  var saveDir: File? = null

  init {
    viewModelScope.launch { checkModel() }
  }

  private suspend fun checkModel() {
    val state = store.inspect(entry)
    _ui.update {
      it.copy(
        model =
          when (state.status) {
            DownloadStatus.READY -> ModelState.Ready
            DownloadStatus.PAUSED -> ModelState.Paused(state.receivedBytes, state.totalBytes)
            else -> ModelState.Missing(entry.totalBytes)
          }
      )
    }
    if (state.status == DownloadStatus.READY) startEngine()
  }

  /**
   * Side-loads a copy from the app's external files dir when one matches the catalog (development
   * only), else downloads the model file (a partial file resumes), then makes the Engine.
   */
  fun download() {
    if (downloadJob?.isActive == true) return
    downloadJob =
      viewModelScope.launch {
        try {
          val app = getApplication<Application>()
          val loaded = store.sideLoad(entry, app.getExternalFilesDir(null))
          loaded.skipped.forEach { Log.w(TAG, "side-load skipped: $it") }
          val before = store.inspect(entry)
          if (before.status != DownloadStatus.READY) {
            val free = StatFs(app.filesDir.path).availableBytes
            val missing = entry.totalBytes - before.receivedBytes
            if (!DownloadSafety.hasSpace(missing, free)) {
              _ui.update {
                it.copy(
                  model =
                    ModelState.Error(
                      "The download needs ${gb(DownloadSafety.requiredFreeBytes(missing))} GB " +
                        "free and ${gb(free)} GB is free."
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
          Log.e(TAG, "DOWNLOAD_FAILED ${e.javaClass.simpleName}: ${e.message}")
          _ui.update { it.copy(model = ModelState.Error(Messages.downloadFailed(e))) }
        }
      }
  }

  private fun startEngine() {
    if (engine != null) return
    val app = getApplication<Application>()
    val speech =
      SpeechEngine(store.file(entry, entry.files.first().name), File(app.cacheDir, "litertlm"), memory)
    engine = speech
    stateJob?.cancel()
    _ui.update { it.copy(engine = EngineState.Idle) }
    stateJob =
      viewModelScope.launch { speech.state.collect { s -> _ui.update { it.copy(engine = s) } } }
    if (BuildConfig.DEBUG) {
      Log.i(TAG, "ENGINE_LOAD mem_available=${MemoryCheck.memAvailableBytes()}")
    }
    speech.load()
  }

  /** Tries to make the Engine again after a failure. */
  fun loadAgain() {
    engine?.load() ?: startEngine()
  }

  /**
   * Debug builds: the memory the check asks for (a failure path of the README). The Engine is
   * released and made again under the new check.
   */
  fun setNeededMemory(bytes: Long): Job =
    viewModelScope.launch {
      memory = MemoryCheck(bytes)
      val old = engine ?: return@launch
      engine = null
      stateJob?.cancel()
      old.close()
      if (_ui.value.model is ModelState.Ready) startEngine()
    }

  fun setTask(task: Task) {
    _ui.update { it.copy(task = task) }
  }

  fun setKeywords(text: String) {
    _ui.update { it.copy(keywords = text.take(Prompts.MAX_KEYWORD_CHARS)) }
  }

  /** Debug builds: the recording cap, in seconds (the README's failure path uses the real one). */
  fun setMaxSeconds(seconds: Int) {
    _ui.update { it.copy(maxSeconds = seconds.coerceIn(1, Prompts.MAX_SECONDS)) }
  }

  fun clearResults() {
    _ui.update { it.copy(results = emptyList(), message = null, messageAction = null) }
  }

  fun dismissMessage() {
    _ui.update { it.copy(message = null, messageAction = null) }
  }

  /** The microphone permission was refused; [blocked] = the system no longer asks. */
  fun permissionRefused(blocked: Boolean) {
    Log.i(TAG, "PERMISSION_REFUSED blocked=$blocked")
    _ui.update {
      it.copy(
        message = if (blocked) Messages.PERMISSION_BLOCKED else Messages.PERMISSION_DENIED,
        messageAction = if (blocked) MessageAction.OPEN_SETTINGS else null,
      )
    }
  }

  /** Why Record cannot start now, or null. */
  fun recordBlocker(): String? {
    val ui = _ui.value
    return when (val m = ui.model) {
      ModelState.Ready -> null
      is ModelState.Downloading ->
        "The model is still downloading (${(100 * m.received / maxOf(1, m.total))} %)."
      ModelState.Verifying -> "The model file is being checked; Record works in a moment."
      ModelState.Checking -> "Looking for the model file; try again in a moment."
      else -> Messages.NO_MODEL
    }
  }

  /**
   * Records from the microphone until Stop (or the cap, or the app leaving the screen), then
   * transcribes what was recorded. `RECORD_AUDIO` is granted before this is called.
   */
  fun record() {
    if (recordJob?.isActive == true || transcribeJob?.isActive == true) return
    recordBlocker()?.let { msg ->
      Log.i(TAG, "REC_REFUSED $msg")
      _ui.update { it.copy(message = msg, messageAction = null) }
      return
    }
    val maxSamples = _ui.value.maxSeconds * Wav.SAMPLE_RATE
    recordJob =
      viewModelScope.launch {
        _ui.update {
          it.copy(recording = true, recordedMs = 0, level = 0f, message = null, messageAction = null)
        }
        Log.i(TAG, "REC_STARTED max_s=${maxSamples / Wav.SAMPLE_RATE}")
        val recording =
          try {
            var lastUi = 0L
            withContext(Dispatchers.IO) {
              recorder.capture(maxSamples) { samples, peak ->
                val now = System.nanoTime()
                if (now - lastUi >= 100_000_000L) {
                  lastUi = now
                  _ui.update { it.copy(recordedMs = samples * 1000L / Wav.SAMPLE_RATE, level = peak) }
                }
              }
            }
          } catch (e: CancellationException) {
            throw e
          } catch (e: Exception) {
            Log.e(TAG, "REC_FAILED ${e.javaClass.simpleName}: ${e.message}")
            _ui.update { it.copy(message = e.message ?: "The microphone did not record.") }
            return@launch
          } finally {
            _ui.update { it.copy(recording = false, level = 0f) }
          }
        Log.i(
          TAG,
          "REC_STOPPED reason=${recording.reason} seconds=${"%.2f".format(recording.seconds)} " +
            "peak=${"%.4f".format(recording.peak)}",
        )
        val note =
          when (recording.reason) {
            StopReason.CAP -> Messages.capReached(maxSamples / Wav.SAMPLE_RATE)
            StopReason.BACKGROUND -> Messages.BACKGROUND
            StopReason.USER -> null
          }
        afterCapture(recording.samples, recording.peak, "mic", note, record = null)?.join()
      }
  }

  fun stopRecording() {
    recorder.stop(StopReason.USER)
  }

  /** The app left the screen: a recording in progress ends and is transcribed. */
  fun onBackground() {
    if (_ui.value.recording) recorder.stop(StopReason.BACKGROUND)
  }

  /** Runs the last recording again with the task and keywords now on the screen. */
  fun again() {
    val samples = lastRecording ?: return
    if (recordJob?.isActive == true || transcribeJob?.isActive == true) return
    transcribe(samples, "again", record = null)
  }

  /**
   * The path every recording takes once it is captured (also the debug plan's files): the checks,
   * then the model. [note] is a sentence to show with the result. Returns the job, or null when
   * nothing went to the model.
   */
  fun afterCapture(
    samples: FloatArray,
    peak: Float,
    source: String,
    note: String?,
    record: JSONObject?,
    cancelAfterMs: Long? = null,
  ): Job? {
    val seconds = samples.size / Wav.SAMPLE_RATE.toDouble()
    saveRecording(samples)
    val problem = RecordingChecks.problem(seconds, peak)
    record?.put("seconds", seconds)
    record?.put("peak", peak.toDouble())
    if (problem != null) {
      Log.i(TAG, "REC_CHECK_REFUSED seconds=${"%.2f".format(seconds)} peak=$peak")
      record?.put("refused", problem)
      _ui.update { it.copy(message = problem, messageAction = null) }
      return null
    }
    val from = SpeechTrim.start(samples)
    if (from == null) {
      Log.i(TAG, "REC_CHECK_REFUSED seconds=${"%.2f".format(seconds)} peak=$peak no_speech")
      record?.put("refused", RecordingChecks.NO_SPEECH)
      _ui.update { it.copy(message = RecordingChecks.NO_SPEECH, messageAction = null) }
      return null
    }
    val speech = if (from == 0) samples else samples.copyOfRange(from, samples.size)
    val sentFrom = from / Wav.SAMPLE_RATE.toDouble()
    Log.i(TAG, "REC_TRIM from_s=${"%.2f".format(sentFrom)} of_s=${"%.2f".format(seconds)}")
    record?.put("sent_from_s", sentFrom)
    lastRecording = speech
    _ui.update { it.copy(hasLastRecording = true, message = note, messageAction = null) }
    return transcribe(speech, source, record, cancelAfterMs)
  }

  private fun transcribe(
    samples: FloatArray,
    source: String,
    record: JSONObject?,
    cancelAfterMs: Long? = null,
  ): Job? {
    val speech = engine
    if (speech == null) {
      _ui.update { it.copy(message = Messages.NO_MODEL) }
      return null
    }
    if (transcribeJob?.isActive == true) return null
    val ui0 = _ui.value
    val task = ui0.task
    val keywords = Prompts.keywords(ui0.keywords)
    val prompt = Prompts.prompt(task, keywords)
    val seconds = samples.size / Wav.SAMPLE_RATE.toDouble()
    val maxOut = Prompts.maxOutputTokens(seconds, prompt)
    val id = nextId++
    val asked = System.nanoTime()
    started = false
    val job =
      viewModelScope.launch {
        put(Transcript(id, task, keywords, seconds, "", done = false))
        _ui.update { it.copy(transcribing = true, stopping = false) }
        val text = StringBuilder()
        var firstMs: Long? = null
        var tokens: Int? = null
        var cancelled = false
        var error: String? = null
        val memBefore = MemoryCheck.memAvailableBytes()
        val stopper =
          cancelAfterMs?.let { ms ->
            launch {
              delay(ms)
              cancel()
            }
          }
        Log.i(
          TAG,
          "TRANSCRIBE_START id=$id source=$source task=${task.name} keywords=${keywords.size} " +
            "seconds=${"%.2f".format(seconds)} max_out=$maxOut",
        )
        try {
          val wav = withContext(Dispatchers.Default) { Wav.encodeFloat(samples) }
          speech.transcribe(wav, prompt, maxOut).collect { e ->
            when (e) {
              is SpeechEvent.Waiting -> Log.i(TAG, "TRANSCRIBE_WAITING id=$id")
              is SpeechEvent.Started -> {
                started = true
                record?.put("wait_ms", e.waitMs)
              }
              is SpeechEvent.Text -> {
                if (firstMs == null) {
                  firstMs = ms(System.nanoTime() - asked)
                  Log.i(TAG, "FIRST_TEXT id=$id ms=$firstMs")
                }
                text.append(e.text)
                put(Transcript(id, task, keywords, seconds, text.toString(), false, firstTextMs = firstMs))
              }
              is SpeechEvent.Done -> tokens = e.tokens
              SpeechEvent.Cancelled -> cancelled = true
            }
          }
        } catch (e: CancellationException) {
          // Stop before the message went in (the Engine was still loading), or the screen closed.
          cancelled = true
          throw e
        } catch (e: Exception) {
          error = Messages.transcribeFailed(e)
          Log.e(TAG, "TRANSCRIBE_FAILED id=$id ${e.javaClass.simpleName}: ${e.message}")
          record?.put("raw_error", "${e.javaClass.simpleName}: ${e.message}")
        } finally {
          stopper?.cancel()
          withContext(NonCancellable) {
            val total = ms(System.nanoTime() - asked)
            val final = text.toString().trim()
            put(
              Transcript(
                id,
                task,
                keywords,
                seconds,
                final,
                done = true,
                cancelled = cancelled,
                firstTextMs = firstMs,
                totalMs = total,
                error = error,
              )
            )
            _ui.update {
              it.copy(
                transcribing = false,
                stopping = false,
                message = if (cancelled) Messages.CANCELLED else error ?: it.message,
              )
            }
            Log.i(
              TAG,
              "TRANSCRIBE_DONE id=$id first_text_ms=$firstMs total_ms=$total chars=${final.length} " +
                "tokens=$tokens cancelled=$cancelled error=${error != null}",
            )
            record?.apply {
              put("id", id)
              put("source", source)
              put("task", task.name)
              put("keywords", keywords.joinToString(", "))
              put("prompt", prompt)
              put("max_output_tokens", maxOut)
              put("text", final)
              put("first_text_ms", firstMs ?: JSONObject.NULL)
              put("total_ms", total)
              put("tokens", tokens ?: JSONObject.NULL)
              put("cancelled", cancelled)
              put("error", error ?: JSONObject.NULL)
              put("mem_available_before", memBefore ?: JSONObject.NULL)
              put("mem_available_after", MemoryCheck.memAvailableBytes() ?: JSONObject.NULL)
              put("vm_hwm_kb", vmHwmKb() ?: JSONObject.NULL)
            }
            saveText(id, source, final, firstMs, total, error)
          }
        }
      }
    transcribeJob = job
    return job
  }

  /** Stops the transcription in flight (the text so far stays). */
  fun cancel() {
    if (!_ui.value.transcribing) return
    _ui.update { it.copy(stopping = true) }
    Log.i(TAG, "TRANSCRIBE_CANCEL_TAPPED started=$started")
    // Before the message went in there is nothing for the runtime to stop: end the wait itself.
    if (started) engine?.cancel() else transcribeJob?.cancel()
  }

  private fun put(t: Transcript) {
    _ui.update { s ->
      val rest = s.results.filter { it.id != t.id }
      s.copy(results = (listOf(t) + rest).take(MAX_RESULTS))
    }
  }

  /** Debug builds with `save_audio`: each recording as a WAV, for the checks in the README. */
  private fun saveRecording(samples: FloatArray) {
    val dir = saveDir ?: return
    runCatching {
      dir.mkdirs()
      File(dir, "rec_${nextId}.wav").writeBytes(Wav.encodeFloat(samples))
    }
  }

  private fun saveText(id: Int, source: String, text: String, firstMs: Long?, total: Long, error: String?) {
    val dir = saveDir ?: return
    runCatching {
      val line =
        JSONObject()
          .put("id", id)
          .put("source", source)
          .put("text", text)
          .put("first_text_ms", firstMs ?: JSONObject.NULL)
          .put("total_ms", total)
          .put("error", error ?: JSONObject.NULL)
      File(dir, "texts.jsonl").appendText(line.toString() + "\n")
    }
  }

  override fun onCleared() {
    recorder.stop(StopReason.BACKGROUND)
    val speech = engine ?: return
    engine = null
    // Not on the main thread: closing waits for a transcription in flight to stop (10 s at most).
    kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch { speech.close() }
  }

  companion object {
    const val TAG = "VoiceToText"
    private const val MAX_RESULTS = 20

    private fun gb(bytes: Long) = "%.1f".format(bytes / 1e9)

    private fun ms(nanos: Long) = nanos / 1_000_000

    /** The peak of [samples], 0..1. */
    fun peak(samples: FloatArray): Float {
      var p = 0f
      for (s in samples) p = maxOf(p, abs(s))
      return p
    }

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
