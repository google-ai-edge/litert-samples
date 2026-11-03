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

package com.google.ai.edge.examples.math_to_latex

import android.app.Application
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.StatFs
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.examples.math_to_latex.data.DownloadSafety
import com.google.ai.edge.examples.math_to_latex.data.DownloadStatus
import com.google.ai.edge.examples.math_to_latex.data.ModelCatalog
import com.google.ai.edge.examples.math_to_latex.data.ModelStore
import java.io.File
import java.security.MessageDigest
import kotlin.math.roundToInt
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
import org.json.JSONArray
import org.json.JSONObject

/** What the model is asked to read in the box. */
enum class Mode(val label: String, val prompt: String, val maxOutputTokens: Int) {
  /** A display formula: the answer is LaTeX. */
  EQUATION("Equation", "Formula Recognition:", 512),

  /** A passage: plain text, with any formula in it as inline LaTeX. */
  TEXT("Text with math", "Text Recognition:", 2048),
}

/** The model file's state, from the first start to a loaded reader. */
sealed class ModelState {
  data object Checking : ModelState()

  data class Missing(val bytes: Long) : ModelState()

  data class Downloading(val received: Long, val total: Long) : ModelState()

  data object Verifying : ModelState()

  data object Ready : ModelState()

  data class Error(val message: String) : ModelState()
}

/** The result of one read, as the screen shows it. */
data class Answer(
  val mode: Mode,
  val text: String,
  val done: Boolean,
  /** From the tap on Read to the first piece of the answer, and to its end. */
  val firstTextMs: Long? = null,
  val totalMs: Long? = null,
  val pieces: Int = 0,
  val pictures: Int = 1,
  val error: String? = null,
)

data class UiState(
  val model: ModelState = ModelState.Checking,
  val reader: ReaderState = ReaderState.Idle,
  val photo: Bitmap? = null,
  val photoName: String? = null,
  /** The box as fractions of the photo (left, top, right, bottom); null = the whole photo. */
  val box: RectF? = null,
  val mode: Mode = Mode.EQUATION,
  val reading: Boolean = false,
  /** Stop was tapped and the read has not ended yet (see [OcrEngine.cancel]). */
  val stopping: Boolean = false,
  val elapsedMs: Long = 0,
  val answer: Answer? = null,
  val message: String? = null,
)

class MathViewModel(app: Application) : AndroidViewModel(app) {
  private val catalog =
    ModelCatalog.parse(app.assets.open("models.json").bufferedReader().use { it.readText() })
  private val entry = catalog.entry(ModelCatalog.OCR)
  private val store = ModelStore(File(app.filesDir, "models"))
  private val _ui = MutableStateFlow(UiState())
  val ui: StateFlow<UiState> = _ui.asStateFlow()

  private var engine: OcrEngine? = null
  private var memory = MemoryCheck()
  private var photoRgb: Rgb? = null
  private var photoSha: String? = null
  private var readJob: Job? = null
  private var downloadJob: Job? = null
  private var stateJob: Job? = null

  init {
    viewModelScope.launch { checkModel() }
  }

  private suspend fun checkModel() {
    val state = store.inspect(entry)
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
   * downloads the model file, then makes the first Engine.
   */
  fun download() {
    if (downloadJob?.isActive == true) return
    downloadJob =
      viewModelScope.launch {
        try {
          val app = getApplication<Application>()
          val loaded = store.sideLoad(entry, app.getExternalFilesDir(null))
          loaded.skipped.forEach { Log.w(TAG, "side-load skipped: $it") }
          if (store.inspect(entry).status != DownloadStatus.READY) {
            val free = StatFs(app.filesDir.path).availableBytes
            val missing = entry.totalBytes - store.inspect(entry).receivedBytes
            if (!DownloadSafety.hasSpace(missing, free)) {
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
    val reader =
      OcrEngine(store.file(entry, entry.files.first().name), File(app.cacheDir, "litertlm"), memory)
    engine = reader
    stateJob?.cancel()
    _ui.update { it.copy(reader = ReaderState.Idle) }
    stateJob =
      viewModelScope.launch { reader.state.collect { s -> _ui.update { it.copy(reader = s) } } }
    reader.prepareNext()
  }

  /**
   * Debug builds: the memory the check asks for (the failure path of the README). The prepared
   * Engine is released and made again under the new check.
   */
  fun setNeededMemory(bytes: Long): Job =
    viewModelScope.launch {
      memory = MemoryCheck(bytes)
      val old = engine ?: return@launch
      engine = null
      stateJob?.cancel()
      old.close()
      startEngine()
    }

  /** Shows the picture in [bytes]; the box and the last answer are cleared. */
  fun setPhoto(bytes: ByteArray, name: String): Job =
    viewModelScope.launch {
      try {
        val (rgb, screen) =
          withContext(Dispatchers.Default) { Photo.decode(bytes) to Photo.decodeForScreen(bytes) }
        photoRgb = rgb
        photoSha = sha256(bytes)
        _ui.update {
          it.copy(photo = screen, photoName = name, box = null, answer = null, message = null)
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

  /** A new box is a new question: the last answer goes. */
  fun setBox(box: RectF?) {
    _ui.update { it.copy(box = box, answer = if (it.reading) it.answer else null) }
  }

  fun setMode(mode: Mode) {
    _ui.update { it.copy(mode = mode) }
  }

  fun dismissMessage() {
    _ui.update { it.copy(message = null) }
  }

  /**
   * The box (fractions of the picture) in the decoded picture's pixels; null = the whole picture.
   */
  private fun pixels(rgb: Rgb, box: RectF?): IntArray {
    if (box == null) return intArrayOf(0, 0, rgb.width, rgb.height)
    val x0 = (box.left * rgb.width).roundToInt().coerceIn(0, rgb.width - 1)
    val y0 = (box.top * rgb.height).roundToInt().coerceIn(0, rgb.height - 1)
    val x1 = (box.right * rgb.width).roundToInt().coerceIn(x0 + 1, rgb.width)
    val y1 = (box.bottom * rgb.height).roundToInt().coerceIn(y0 + 1, rgb.height)
    return intArrayOf(x0, y0, x1, y1)
  }

  /** Reads the box (or the whole photo) with the chosen mode. */
  fun read() {
    readOnce(record = null)
  }

  /**
   * One read; [record] (debug runs) gets the details. [pixelBox] (debug) is a box in the decoded
   * picture's pixels, which replaces the screen box. Returns the job.
   */
  fun readOnce(
    record: JSONObject?,
    pixelBox: IntArray? = null,
    renderPrompt: Boolean = false,
    cancelAfterMs: Long? = null,
  ): Job? {
    val reader = engine ?: return null
    val rgb = photoRgb ?: return null
    if (readJob?.isActive == true) return null
    val ui0 = _ui.value
    val mode = ui0.mode
    val tapped = System.nanoTime()
    val job =
      viewModelScope.launch {
        _ui.update {
          it.copy(
            reading = true,
            stopping = false,
            elapsedMs = 0,
            answer = Answer(mode, "", done = false),
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
        var firstMs: Long? = null
        var pieces = 0
        var pictures = 1
        val memBefore = MemoryCheck.memAvailableBytes()
        var stopper: Job? = null
        try {
          val prepared =
            withContext(Dispatchers.Default) {
              val b = pixelBox ?: pixels(rgb, ui0.box)
              val p = Pictures.prepare(rgb.crop(b[0], b[1], b[2], b[3]))
              val shas = p.pictures.map { it.sha256() }
              // Debug builds: coordinates and hashes only, enough to give a reference run the same
              // pictures.
              if (BuildConfig.DEBUG)
                Log.i(
                  TAG,
                  "READ_INPUT mode=${mode.name} photo=$photoSha box=${b.joinToString(",")} " +
                    "pictures=${p.pictures.size} slice_cols=${p.sliceCols} " +
                    "rgb_sha256=${shas.joinToString(",") { it.take(16) }}",
                )
              record?.put("pixel_box", JSONArray(b.toList()))
              record?.put("pictures_rgb_sha256", JSONArray(shas))
              p to p.pictures.map { Photo.png(it) }
            }
          pictures = prepared.first.pictures.size
          record?.put("pictures", pictures)
          record?.put("slice_cols", prepared.first.sliceCols ?: JSONObject.NULL)
          record?.put("prepare_ms", (System.nanoTime() - tapped) / 1_000_000)
          stopper =
            cancelAfterMs?.let { ms ->
              launch {
                delay(ms)
                reader.cancel()
              }
            }
          reader
            .read(
              prepared.second,
              mode.prompt,
              prepared.first.extraContext,
              mode.maxOutputTokens,
              renderPrompt,
            )
            .collect { e ->
              when (e) {
                is ReadEvent.Waiting -> _ui.update { it.copy(message = null) }
                is ReadEvent.Rendered -> record?.put("rendered_prompt", e.prompt)
                is ReadEvent.Started -> {
                  record?.put("wait_ms", e.waitMs)
                  record?.put("engine_ms", e.engineMs)
                  record?.put("conversation_ms", e.conversationMs)
                }
                is ReadEvent.Text -> {
                  if (firstMs == null) firstMs = (System.nanoTime() - tapped) / 1_000_000
                  pieces++
                  text.append(e.text)
                  _ui.update {
                    it.copy(
                      answer =
                        Answer(
                          mode,
                          text.toString(),
                          done = false,
                          firstTextMs = firstMs,
                          pieces = pieces,
                          pictures = pictures,
                        )
                    )
                  }
                }
              }
            }
          val total = (System.nanoTime() - tapped) / 1_000_000
          _ui.update {
            it.copy(
              answer =
                Answer(
                  mode,
                  text.toString().trim(),
                  done = true,
                  firstTextMs = firstMs,
                  totalMs = total,
                  pieces = pieces,
                  pictures = pictures,
                )
            )
          }
          record?.put("first_text_ms", firstMs ?: JSONObject.NULL)
          record?.put("total_ms", total)
          Log.i(
            TAG,
            "READ mode=${mode.name} pictures=$pictures first_text_ms=$firstMs " +
              "total_ms=$total pieces=$pieces",
          )
        } catch (e: CancellationException) {
          throw e
        } catch (e: Exception) {
          Log.e(TAG, "READ_FAILED ${e.javaClass.simpleName}: ${e.message}")
          record?.put("error", "${e.javaClass.simpleName}: ${e.message}")
          _ui.update {
            it.copy(
              answer =
                Answer(
                  mode,
                  text.toString(),
                  done = true,
                  pieces = pieces,
                  error = e.message ?: e.javaClass.simpleName,
                )
            )
          }
        } finally {
          ticker.cancel()
          stopper?.cancel()
          withContext(NonCancellable) {
            record?.put("mode", mode.name)
            record?.put("prompt", mode.prompt)
            record?.put("photo_sha256", photoSha)
            record?.put("text", text.toString())
            record?.put("pieces", pieces)
            record?.put("mem_available_before", memBefore ?: JSONObject.NULL)
            record?.put("mem_available_after", MemoryCheck.memAvailableBytes() ?: JSONObject.NULL)
            record?.put("vm_hwm_kb", vmHwmKb() ?: JSONObject.NULL)
            _ui.update { it.copy(reading = false, stopping = false) }
          }
        }
      }
    readJob = job
    return job
  }

  fun stop() {
    if (!_ui.value.reading) return
    _ui.update { it.copy(stopping = true) }
    engine?.cancel()
  }

  override fun onCleared() {
    val reader = engine ?: return
    engine = null
    // Not on the main thread: closing waits for a read in flight to stop (10 s at most).
    kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch { reader.close() }
  }

  companion object {
    const val TAG = "MathToLatex"

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
