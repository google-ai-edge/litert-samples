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

package com.google.ai.edge.examples.zero_shot_classification

import android.content.Context
import android.net.ConnectivityManager
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the engine and confines all tokenizer, builder, model, and decoder calls to one dispatcher.
 */
class MainViewModel(private val context: Context) : ViewModel() {
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(SupervisorJob() + modelDispatcher)
  private val preferences = context.getSharedPreferences("laya_preferences", Context.MODE_PRIVATE)
  private var engine: LayaEngine? = null
  private var presets: Map<String, Any?>? = null
  private var started = false
  // The downloaded manifest holds the float16-weight graph only.
  private val storage = LayaEngine.Storage.WFP16
  private var launchStartedNs = 0L
  private val downloader by lazy {
    ModelDownloader(
      context.assets.open("model_manifest.json").bufferedReader().use { it.readText() },
      context.filesDir,
    )
  }
  private var downloadJob: Job? = null
  @Volatile private var cleared = false
  private val npuAvailable = LayaEngine.isNpuAvailable(context)
  private val mutableUiState =
    MutableStateFlow(
      UiState(
        inputSubject = context.getString(R.string.example_email_en_subject),
        inputText = context.getString(R.string.example_email_en_body),
        accelerator = if (npuAvailable) LayaEngine.Backend.NPU else LayaEngine.Backend.GPU,
        npuAvailable = npuAvailable,
      )
    )

  /** State consumed by the screen with lifecycle-aware collection. */
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()

  /** Starts the interactive pipeline once per ViewModel lifetime. */
  fun start(accelerator: String?, launchedAtNs: Long = System.nanoTime()) {
    if (started || cleared) return
    started = true
    launchStartedNs = launchedAtNs
    // An NPU start that never reached Ready (the process died while compiling) falls back to the
    // GPU, and so does a saved NPU choice from an install that had the NPU runtime.
    val npuUsable = npuAvailable && !preferences.getBoolean(NPU_PENDING, false)
    preferences.edit().remove(NPU_PENDING).apply()
    val default = if (npuUsable) "npu" else "gpu"
    val saved =
      (preferences.getString("accelerator", default) ?: default).let {
        if (it == "npu" && !npuUsable) "gpu" else it
      }
    val backend =
      try {
        LayaEngine.backendFromArgument(accelerator ?: saved)
      } catch (failure: IllegalStateException) {
        showFailure(failure)
        return
      }
    loadAccelerator(backend)
  }

  /**
   * Downloads or resumes the model files after the user taps the button, then loads them. The
   * files come from the pinned revision in model_manifest.json and are checked by SHA-256.
   */
  fun downloadModelFiles() {
    if (cleared || !uiState.value.downloadNeeded || !downloading.compareAndSet(false, true)) return
    val startedNs = System.nanoTime()
    mutableUiState.update {
      it.copy(
        downloading = true,
        downloadFailed = false,
        spaceNeededBytes = null,
        spaceAvailableBytes = null,
        errorMessage = null,
        statusMessage = R.string.status_downloading,
      )
    }
    downloadJob =
      viewModelScope.launch(Dispatchers.IO) {
        try {
          var shownBytes = -1L
          downloader.download { done, total ->
            ensureActive()
            if (done == total || done - shownBytes >= PROGRESS_STEP_BYTES) {
              shownBytes = done
              mutableUiState.update { it.copy(downloadedBytes = done, downloadTotalBytes = total) }
            }
          }
          Log.i(
            "LAYA_DOWNLOAD",
            "total_bytes=${downloader.totalBytes} " +
              "elapsed_ms=${milliseconds(System.nanoTime() - startedNs)}",
          )
          mutableUiState.update { it.copy(downloadNeeded = false, downloading = false) }
          if (cleared) return@launch
          loadAccelerator(uiState.value.accelerator)
        } catch (failure: ModelDownloader.InsufficientSpaceException) {
          Log.w(
            "LAYA_DOWNLOAD",
            "Not enough free space: needed=${failure.neededBytes} " +
              "available=${failure.availableBytes}",
          )
          mutableUiState.update {
            it.copy(
              downloading = false,
              spaceNeededBytes = failure.neededBytes,
              spaceAvailableBytes = failure.availableBytes,
              statusMessage = R.string.status_not_downloaded,
            )
          }
        } catch (failure: CancellationException) {
          throw failure
        } catch (failure: Exception) {
          Log.w("LAYA_DOWNLOAD", "Download failed", failure)
          showDownloadNeeded(failure.message ?: failure.javaClass.simpleName)
        } finally {
          downloading.set(false)
        }
      }
  }

  /** Edits the email subject without changing the preset schema. */
  fun setSubject(value: String) {
    if (!editable()) return
    mutableUiState.update { it.copy(inputSubject = value).withoutResults() }
  }

  /** Edits the email body, support message, or moderation post. */
  fun setInputText(value: String) {
    if (!editable()) return
    mutableUiState.update { it.copy(inputText = value).withoutResults() }
  }

  /** Replaces the editor contents with this preset's invented example in the selected language. */
  fun selectLanguage(value: ExampleLanguage) {
    if (!editable()) return
    mutableUiState.update { withExample(it.copy(language = value)).withoutResults() }
  }

  /** Selects the unchanged upstream questions and their matching invented example. */
  fun selectPreset(value: Preset) {
    if (!editable()) return
    mutableUiState.update { withExample(it.copy(preset = value)).withoutResults() }
  }

  /** Saves an explicit backend choice; a compile failure never selects another one. */
  fun selectAccelerator(value: LayaEngine.Backend) {
    if (!editable() || (value == uiState.value.accelerator && uiState.value.ready)) return
    loadAccelerator(value)
  }

  /** Selects the calibration JSON or the unchanged decoder's identity temperature. */
  fun setCalibrated(value: Boolean) {
    if (!editable()) return
    mutableUiState.update { it.copy(calibrated = value).withoutResults() }
  }

  private fun editable() = !cleared && !uiState.value.busy && !uiState.value.downloadNeeded

  private fun loadAccelerator(backend: LayaEngine.Backend) {
    val initializationStartedNs = System.nanoTime()
    preferences.edit().putString("accelerator", backend.name.lowercase()).apply()
    mutableUiState.update {
      it
        .withoutResults()
        .copy(
          accelerator = backend,
          busy = true,
          ready = false,
          errorMessage = null,
          fallback = null,
          statusMessage = R.string.status_loading_tokenizer,
        )
    }
    modelScope.launch {
      var compiling = false
      try {
        if (!downloader.isComplete()) {
          showDownloadNeeded()
          return@launch
        }
        val helper = helper()
        mutableUiState.update {
          it.copy(
            statusMessage =
              when (backend) {
                LayaEngine.Backend.NPU -> R.string.status_compiling_npu
                LayaEngine.Backend.GPU -> R.string.status_compiling_gpu
                LayaEngine.Backend.CPU -> R.string.status_loading_cpu
              }
          )
        }
        compiling = true
        if (backend == LayaEngine.Backend.NPU) {
          preferences.edit().putBoolean(NPU_PENDING, true).commit()
        }
        val compileStartedNs = System.nanoTime()
        helper.initialize(backend)
        val compileMs = milliseconds(System.nanoTime() - compileStartedNs)
        compiling = false
        mutableUiState.update { it.copy(statusMessage = R.string.status_warming_up) }
        val warmStartedNs = System.nanoTime()
        val state = uiState.value
        // Preserve the full preset warm-up, including tokenization, building, both graphs and
        // decode.
        questions(state.preset).forEach { (id, question) ->
          answer(helper, state, id, LayaJson.asObject(question))
        }
        preferences.edit().remove(NPU_PENDING).apply()
        val readyAtNs = System.nanoTime()
        val warmupMs = milliseconds(readyAtNs - warmStartedNs)
        val launchToReadyMs =
          uiState.value.launchToReadyMs ?: milliseconds(readyAtNs - launchStartedNs)
        mutableUiState.update {
          it.copy(
            busy = false,
            ready = true,
            launchToReadyMs = launchToReadyMs,
            statusMessage = readyStatus(backend),
          )
        }
        Log.i(
          "LAYA_READY",
          "accelerator=${backend.name.lowercase()} storage=${storage.argument} " +
            "launch_to_ready_ms=$launchToReadyMs " +
            "initialization_ms=${milliseconds(readyAtNs - initializationStartedNs)} " +
            "tokenizer_load_ms=${helper.tokenizerLoadMs} " +
            "embedding_map_ms=${helper.embeddingLoadMs} compile_ms=$compileMs warmup_ms=$warmupMs",
        )
      } catch (failure: Exception) {
        preferences.edit().remove(NPU_PENDING).apply()
        showFailure(failure, fallback = if (compiling) fallbackFor(backend) else null)
      } catch (failure: LinkageError) {
        preferences.edit().remove(NPU_PENDING).apply()
        showFailure(failure, fallback = if (compiling) fallbackFor(backend) else null)
      }
    }
  }

  /**
   * Runs the current preset and reports complete host-plus-model time, excluding initialization.
   */
  fun run() {
    val state = uiState.value
    if (!editable() || !state.ready) return
    val runStartedNs = System.nanoTime()
    mutableUiState.update {
      it
        .withoutResults()
        .copy(busy = true, errorMessage = null, statusMessage = R.string.status_running)
    }
    modelScope.launch {
      try {
        val results = mutableListOf<AnswerUiRow>()
        questions(state.preset).forEach { (id, question) ->
          val output = answer(helper(), state, id, LayaJson.asObject(question))
          results += presentAnswer(id, output, state.calibrated)
          mutableUiState.update { it.copy(answers = results.toList()) }
        }
        val runTotalMs = milliseconds(System.nanoTime() - runStartedNs)
        val tokenCount = results.sumOf { it.tokenCount }
        mutableUiState.update {
          it.copy(
            busy = false,
            runTotalMs = runTotalMs,
            runTokenCount = tokenCount,
            statusMessage = readyStatus(state.accelerator),
          )
        }
        Log.i(
          "LAYA_TIMING",
          "accelerator=${state.accelerator.name.lowercase()} storage=${storage.argument} " +
            "total_ms=$runTotalMs question_count=${results.size} token_count=$tokenCount " +
            "calibrated=${state.calibrated}",
        )
      } catch (failure: Exception) {
        showFailure(failure)
      } catch (failure: LinkageError) {
        showFailure(failure)
      }
    }
  }

  private fun answer(
    helper: LayaEngine,
    state: UiState,
    id: String,
    question: Map<String, Any?>,
  ): LayaEngine.AnswerResult {
    val startedNs = System.nanoTime()
    val sequence = helper.prepare(modelState(state), question, questionId = id)
    val preparedNs = System.nanoTime()
    val raw = helper.runRaw(sequence, state.accelerator)
    check(raw.finite) { context.getString(R.string.error_nonfinite) }
    val decodeStartedNs = System.nanoTime()
    val calibration = if (state.calibrated) helper.calibration else LayaCalibration.identity()
    val answer = LayaDecoder.decode(raw.markerLogits, raw.actLogits, sequence.question, calibration)
    val finishedNs = System.nanoTime()
    return LayaEngine.AnswerResult(
      sequence,
      answer,
      raw,
      milliseconds(preparedNs - startedNs),
      milliseconds(finishedNs - decodeStartedNs),
      milliseconds(finishedNs - startedNs),
    )
  }

  private fun presentAnswer(
    id: String,
    output: LayaEngine.AnswerResult,
    calibrated: Boolean,
  ): AnswerUiRow {
    val answer = output.answer
    val question = output.sequence.question
    val probabilities =
      when (question.type) {
        "choice" -> {
          val criteria = LayaJson.asObject(question.criteria)
          LayaJson.asObject(answer.getValue("probabilities")).map { (label, value) ->
            ProbabilityUiRow(
              probability = (value as Number).toDouble(),
              label = label,
              description = criteria[label]?.let(LayaPromptBuilder::renderCriterion),
            )
          }
        }
        "score" -> {
          val legend = LayaJson.asObject(answer.getValue("legend"))
          LayaJson.asObject(answer.getValue("probabilities")).map { (level, value) ->
            ProbabilityUiRow(
              probability = (value as Number).toDouble(),
              scoreLevel = level.toInt(),
              description = legend[level]?.let(LayaPromptBuilder::renderCriterion),
            )
          }
        }
        else -> {
          val calibration = if (calibrated) helper().calibration else LayaCalibration.identity()
          val values = LayaDecoder.probabilities(output.raw.markerLogits, question, calibration)
          listOf(
            ProbabilityUiRow(
              probability = LayaDecoder.round4(values[0].toDouble()),
              labelResource = R.string.option_false,
            ),
            ProbabilityUiRow(
              probability = LayaDecoder.round4(values[1].toDouble()),
              labelResource = R.string.option_true,
            ),
          )
        }
      }
    return AnswerUiRow(
      questionId = id,
      instructions = question.instructions,
      type = question.type,
      choice = answer["choice"] as? String,
      score = (answer["score"] as? Number)?.toDouble(),
      trueProbability = (answer["noul"] as? Number)?.toDouble(),
      probabilities = probabilities,
      confidence = (answer.getValue("confidence") as Number).toDouble(),
      totalMs = output.totalMs,
      tokenCount = output.sequence.ids.size,
    )
  }

  private fun modelState(state: UiState): Map<String, String> =
    when (state.preset) {
      Preset.EMAIL -> linkedMapOf("subject" to state.inputSubject, "body" to state.inputText)
      Preset.TRIAGE -> linkedMapOf("message" to state.inputText)
      Preset.MODERATION -> linkedMapOf("post" to state.inputText)
    }

  private fun withExample(state: UiState): UiState {
    val japanese = state.language == ExampleLanguage.JA
    val subject =
      if (state.preset == Preset.EMAIL) {
        context.getString(
          if (japanese) R.string.example_email_ja_subject else R.string.example_email_en_subject
        )
      } else {
        ""
      }
    val text =
      when (state.preset) {
        Preset.EMAIL ->
          if (japanese) R.string.example_email_ja_body else R.string.example_email_en_body
        Preset.TRIAGE -> if (japanese) R.string.example_support_ja else R.string.example_support_en
        Preset.MODERATION ->
          if (japanese) R.string.example_moderation_ja else R.string.example_moderation_en
      }
    return state.copy(inputSubject = subject, inputText = context.getString(text))
  }

  private fun questions(preset: Preset): Map<String, Any?> {
    val all =
      presets
        ?: context.assets
          .open("presets.json")
          .bufferedReader()
          .use { LayaJson.asObject(LayaJson.parse(it.readText())) }
          .also { presets = it }
    return LayaJson.asObject(all.getValue(preset.assetKey))
  }

  private fun helper(): LayaEngine = engine ?: LayaEngine(context, storage).also { engine = it }

  /** Shows the download button, or Retry with [errorMessage] after a failed attempt. */
  private fun showDownloadNeeded(errorMessage: String? = null) {
    mutableUiState.update {
      it.copy(
        busy = false,
        downloadNeeded = true,
        downloading = false,
        downloadedBytes = downloader.bytesDownloaded(),
        downloadTotalBytes = downloader.totalBytes,
        meteredNetwork = isMetered(),
        downloadFailed = errorMessage != null,
        errorMessage = errorMessage,
        statusMessage = R.string.status_not_downloaded,
      )
    }
  }

  private fun isMetered(): Boolean {
    val manager = context.getSystemService(ConnectivityManager::class.java) ?: return false
    // isActiveNetworkMetered is also true when there is no network at all.
    return manager.activeNetwork != null && manager.isActiveNetworkMetered
  }

  private fun showFailure(failure: Throwable, fallback: LayaEngine.Backend? = null) {
    mutableUiState.update {
      it.copy(
        busy = false,
        ready = false,
        statusMessage = R.string.status_error,
        runTotalMs = null,
        errorMessage = failure.message ?: failure.javaClass.simpleName,
        fallback = fallback,
      )
    }
  }

  private fun UiState.withoutResults() =
    copy(answers = emptyList(), runTotalMs = null, runTokenCount = 0)

  override fun onCleared() {
    cleared = true
    // Stops a running download. Its .part files stay, and the next launch resumes them.
    downloadJob?.cancel()
    // Queue cleanup after any blocking native call on the same confined dispatcher.
    modelScope.launch {
      try {
        engine?.close()
        engine = null
      } catch (failure: Exception) {
        Log.e("LAYA", "Engine cleanup failed", failure)
      } finally {
        modelScope.cancel()
      }
    }
    super.onCleared()
  }

  companion object {
    private const val PROGRESS_STEP_BYTES = 1_000_000L

    /** Set while an NPU compile runs; still set at the next launch if the process died in it. */
    private const val NPU_PENDING = "npu_pending"

    // One download per process, so two ViewModel instances never write the same .part file.
    private val downloading = AtomicBoolean(false)

    private fun milliseconds(nanos: Long) = nanos / 1_000_000.0

    private fun readyStatus(backend: LayaEngine.Backend) =
      when (backend) {
        LayaEngine.Backend.NPU -> R.string.status_npu_ready
        LayaEngine.Backend.GPU -> R.string.status_gpu_ready
        LayaEngine.Backend.CPU -> R.string.status_cpu_ready
      }

    private fun fallbackFor(backend: LayaEngine.Backend) =
      when (backend) {
        LayaEngine.Backend.NPU -> LayaEngine.Backend.GPU
        LayaEngine.Backend.GPU -> LayaEngine.Backend.CPU
        LayaEngine.Backend.CPU -> null
      }

    /** Creates a ViewModel with the application context, never an Activity reference. */
    fun getFactory(context: Context): ViewModelProvider.Factory =
      object : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
          require(modelClass.isAssignableFrom(MainViewModel::class.java))
          @Suppress("UNCHECKED_CAST")
          return MainViewModel(context.applicationContext) as T
        }
      }
  }
}
