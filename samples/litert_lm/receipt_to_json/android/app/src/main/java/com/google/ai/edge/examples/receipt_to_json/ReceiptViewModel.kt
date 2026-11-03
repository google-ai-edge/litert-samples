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

package com.google.ai.edge.examples.receipt_to_json

import android.app.Application
import android.graphics.Bitmap
import android.os.StatFs
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.examples.receipt_to_json.data.DownloadSafety
import com.google.ai.edge.examples.receipt_to_json.data.DownloadStatus
import com.google.ai.edge.examples.receipt_to_json.data.ModelCatalog
import com.google.ai.edge.examples.receipt_to_json.data.ModelStore
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
import org.json.JSONArray
import org.json.JSONObject

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
  val text: String,
  val done: Boolean,
  val keys: List<Kie.Key>,
  /** From the tap on Extract to the first piece of the answer, and to its end. */
  val firstTextMs: Long? = null,
  val totalMs: Long? = null,
  val pieces: Int = 0,
  /** How many of [keys] the answer's JSON object holds; null = no JSON object that parses. */
  val keysFound: Int? = null,
  /** The number of rows the answer gives each list key. */
  val rows: List<Pair<String, Int>> = emptyList(),
  val error: String? = null,
)

data class UiState(
  val model: ModelState = ModelState.Checking,
  val reader: ReaderState = ReaderState.Idle,
  val photo: Bitmap? = null,
  val photoName: String? = null,
  /** The keys to extract, as typed: separated by commas or new lines. */
  val keysText: String = Kie.DEFAULT_KEYS.joinToString(", "),
  val reading: Boolean = false,
  /** Stop was tapped and the read has not ended yet. */
  val stopping: Boolean = false,
  val elapsedMs: Long = 0,
  val answer: Answer? = null,
  val message: String? = null,
)

class ReceiptViewModel(app: Application) : AndroidViewModel(app) {
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

  /** Shows the picture in [bytes]; the last answer is cleared. */
  fun setPhoto(bytes: ByteArray, name: String): Job =
    viewModelScope.launch {
      try {
        val (rgb, screen) =
          withContext(Dispatchers.Default) { Photo.decode(bytes) to Photo.decodeForScreen(bytes) }
        photoRgb = rgb
        photoSha = sha256(bytes)
        _ui.update { it.copy(photo = screen, photoName = name, answer = null, message = null) }
        if (BuildConfig.DEBUG)
          Log.i(TAG, "PHOTO name=$name ${rgb.width}x${rgb.height} sha256=$photoSha")
      } catch (e: CancellationException) {
        throw e
      } catch (e: Exception) {
        Log.w(TAG, "PHOTO_FAILED ${e.message}")
        _ui.update { it.copy(message = "Cannot open $name: ${e.message}") }
      }
    }

  fun setKeys(text: String) {
    _ui.update { it.copy(keysText = text) }
  }

  fun dismissMessage() {
    _ui.update { it.copy(message = null) }
  }

  /** Extracts the keys in the editor from the photo. */
  fun extract() {
    readOnce(record = null)
  }

  /**
   * One read; [record] (debug runs) gets the details. [keysText] (debug) replaces the editor's
   * keys. Returns the job, or null when nothing was started.
   */
  fun readOnce(
    record: JSONObject?,
    keysText: String? = null,
    renderPrompt: Boolean = false,
    cancelAfterMs: Long? = null,
  ): Job? {
    val reader = engine ?: return null
    val rgb = photoRgb ?: return null
    if (readJob?.isActive == true) return null
    if (keysText != null) setKeys(keysText)
    val keys =
      try {
        Kie.parseKeys(_ui.value.keysText)
      } catch (e: IllegalArgumentException) {
        _ui.update { it.copy(message = e.message) }
        record?.put("error", e.message)
        return null
      }
    if (keys.isEmpty()) {
      _ui.update { it.copy(message = "Type at least one key, for example total.") }
      record?.put("error", "no keys")
      return null
    }
    val prompt = Kie.prompt(keys)
    val tapped = System.nanoTime()
    val job =
      viewModelScope.launch {
        _ui.update {
          it.copy(
            reading = true,
            stopping = false,
            elapsedMs = 0,
            answer = Answer("", done = false, keys = keys),
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
        val memBefore = MemoryCheck.memAvailableBytes()
        var stopper: Job? = null
        try {
          val png =
            withContext(Dispatchers.Default) {
              val picture = Pictures.padTo(rgb)
              val sha = picture.sha256()
              // Debug builds: the hash of the 768 x 1344 picture, enough to give a reference run the
              // same pixels.
              if (BuildConfig.DEBUG) Log.i(TAG, "READ_INPUT photo=$photoSha picture_rgb_sha256=$sha")
              record?.put("picture_rgb_sha256", sha)
              Photo.png(picture)
            }
          record?.put("prepare_ms", (System.nanoTime() - tapped) / 1_000_000)
          stopper =
            cancelAfterMs?.let { ms ->
              launch {
                delay(ms)
                reader.cancel()
              }
            }
          reader.read(png, prompt, MAX_OUTPUT_TOKENS, renderPrompt).collect { e ->
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
                        text.toString(),
                        done = false,
                        keys = keys,
                        firstTextMs = firstMs,
                        pieces = pieces,
                      )
                  )
                }
              }
            }
          }
          val total = (System.nanoTime() - tapped) / 1_000_000
          val answer = text.toString().trim()
          val found = Kie.keysFound(answer, keys)
          val rows = Kie.rows(answer, keys)
          _ui.update {
            it.copy(
              answer =
                Answer(
                  answer,
                  done = true,
                  keys = keys,
                  firstTextMs = firstMs,
                  totalMs = total,
                  pieces = pieces,
                  keysFound = found,
                  rows = rows,
                )
            )
          }
          record?.put("first_text_ms", firstMs ?: JSONObject.NULL)
          record?.put("total_ms", total)
          record?.put("keys_found", found ?: JSONObject.NULL)
          record?.put("rows", JSONObject(rows.toMap()))
          val rowText = rows.joinToString(" ") { (k, n) -> "$k=$n" }
          Log.i(
            TAG,
            "READ first_text_ms=$firstMs total_ms=$total pieces=$pieces keys_found=$found rows=$rowText",
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
                  text.toString(),
                  done = true,
                  keys = keys,
                  pieces = pieces,
                  error = e.message ?: e.javaClass.simpleName,
                )
            )
          }
        } finally {
          ticker.cancel()
          stopper?.cancel()
          withContext(NonCancellable) {
            record?.put("keys", JSONArray(keys.map { it.toString() }))
            record?.put("prompt", prompt)
            record?.put("photo_sha256", photoSha)
            record?.put("text", text.toString())
            record?.put("pieces", pieces)
            record?.put("max_output_tokens", MAX_OUTPUT_TOKENS)
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
    const val TAG = "ReceiptToJson"

    /**
     * The answer's cap. The file holds 4,096 tokens and a picture with its prompt takes about
     * 1,030-1,100 of them; a receipt's fields take far fewer than this.
     */
    const val MAX_OUTPUT_TOKENS = 1024

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
