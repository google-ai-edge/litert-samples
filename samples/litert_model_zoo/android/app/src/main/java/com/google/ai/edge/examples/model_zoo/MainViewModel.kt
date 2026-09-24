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

package com.google.ai.edge.examples.model_zoo

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.ConnectivityManager
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.examples.model_zoo.audio.AudioPlayback
import com.google.ai.edge.examples.model_zoo.audio.AudioStageProgress
import com.google.ai.edge.examples.model_zoo.audio.AudioTaskEngine
import com.google.ai.edge.examples.model_zoo.audio.AudioTaskRequest
import com.google.ai.edge.examples.model_zoo.audio.BatchAudioTasks
import com.google.ai.edge.examples.model_zoo.audio.MatchaEngine
import com.google.ai.edge.examples.model_zoo.audio.NamedWaveform
import com.google.ai.edge.examples.model_zoo.audio.SpeechOutputCache
import com.google.ai.edge.examples.model_zoo.audio.WavAudio
import com.google.ai.edge.examples.model_zoo.audio.ZipformerEngine
import com.google.ai.edge.examples.model_zoo.common.AudioCapture
import com.google.ai.edge.examples.model_zoo.data.DownloadConfirmation
import com.google.ai.edge.examples.model_zoo.data.DownloadSafety
import com.google.ai.edge.examples.model_zoo.data.DownloadState
import com.google.ai.edge.examples.model_zoo.data.DownloadStatus
import com.google.ai.edge.examples.model_zoo.data.ModelCatalog
import com.google.ai.edge.examples.model_zoo.data.ModelEntry
import com.google.ai.edge.examples.model_zoo.data.ModelStore
import com.google.ai.edge.examples.model_zoo.image.ImageTaskRequest
import com.google.ai.edge.examples.model_zoo.image.RealtimeImageTasks
import com.google.ai.edge.examples.model_zoo.image.SingleImageEngine
import com.google.ai.edge.examples.model_zoo.image.SingleImageTasks
import com.google.ai.edge.examples.model_zoo.vision.DetectionEngine
import com.google.ai.edge.examples.model_zoo.models.zipformer.ZipformerAsr
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.Executors
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

class MainViewModel(application: Application) : AndroidViewModel(application) {
  private val mutable = MutableStateFlow(UiState())
  val state = mutable.asStateFlow()
  private val inferenceDispatcher =
    Executors.newSingleThreadExecutor { Thread(it, "ModelZooInference") }.asCoroutineDispatcher()
  private val cleanupScope = CoroutineScope(SupervisorJob() + inferenceDispatcher)
  private val store = ModelStore(File(application.filesDir, "models"))
  private val downloads = mutableMapOf<String, Job>()
  private val deleting = mutableSetOf<String>()
  private var detector: DetectionEngine? = null
  private var synthesizer: MatchaEngine? = null
  private var recognizer: ZipformerEngine? = null
  private var singleImageEngine: Pair<String, SingleImageEngine>? = null
  private var batchAudioEngine: Pair<String, AudioTaskEngine>? = null
  private var audioWaveforms: List<NamedWaveform> = emptyList()
  private val captureLock = Any()
  private var recordingSession: RecordingSession? = null
  private val player = AudioPlayback()
  private var speech: FloatArray? = null
  private var speechRate = 22050
  private val speechOutput = SpeechOutputCache(application.cacheDir)
  private var speechCached = false
  @Volatile private var cleared = false
  @Volatile private var cameraEnabled = false
  @Volatile private var navigationGeneration = 0L

  private class RecordingSession(val generation: Long, val taskId: String) {
    val capture = AudioCapture(sampleRate = 16000)
    val samples =
      FloatArray(if (taskId == "speech-recognition") ZipformerAsr.MAX_SAMPLES else 16000 * 12)
    var count = 0
  }

  init {
    viewModelScope.launch {
      try {
        val catalog =
          withContext(Dispatchers.IO) {
            application.assets.open("models.json").bufferedReader().use {
              ModelCatalog.parse(it.readText())
            }
          }
        val visibleTasks = catalog.tasks
        mutable.update { it.copy(tasks = visibleTasks) }
        for (entry in visibleTasks) {
          val download = store.inspect(entry)
          mutable.update { it.copy(downloads = it.downloads + (entry.taskId to download)) }
        }
        refreshStorage()
      } catch (e: Throwable) {
        showError(TaskFailures.message(e))
      } finally {
        mutable.update { it.copy(loading = false) }
      }
    }
  }

  fun navigate(screen: String, taskId: String? = null) {
    if (screen == "task" && state.value.tasks.none { it.taskId == taskId }) return
    navigationGeneration++
    cameraEnabled = false
    stopRecording(transcribe = false)
    player.close()
    cleanupScope.launch {
      try {
        singleImageEngine?.second?.close()
        batchAudioEngine?.second?.close()
      } catch (failure: Throwable) {
        showError(TaskFailures.message(failure))
      } finally {
        singleImageEngine = null
        batchAudioEngine = null
      }
    }
    audioWaveforms = emptyList()
    mutable.update {
      it.copy(
        screen = screen,
        selectedTaskId = taskId,
        downloadConfirmation = null,
        camera = false,
        image = null,
        boxes = emptyList(),
        inferenceMs = null,
        backend = "",
        fallbackReason = null,
        backendDetails = "",
        error = null,
        transcript = null,
        inputImage = null,
        outputImage = null,
        imageOutputText = "",
        imageOutputDetails = "",
        secondaryImage = null,
        cameraFrames = 0,
        audioOutputs = emptyList(),
        audioSummary = "",
        audioInputName = null,
        audioInputSeconds = null,
        playing = false,
        playbackElapsedSeconds = 0f,
        playbackTotalSeconds =
          if (taskId == "text-to-speech") (speech?.size ?: 0) / speechRate.toFloat() else 0f,
        audioProgress = null,
        pitchHz = emptyList(),
        pitchConfidence = emptyList(),
        audioReady = taskId == "text-to-speech" && speech != null,
        speechExportReady = taskId == "text-to-speech" && speech != null && speechCached,
        speechSaved = false,
      )
    }
  }

  fun updateText(text: String) {
    mutable.update { it.copy(inputText = text) }
  }

  fun showError(error: String) {
    mutable.update { it.copy(error = error) }
  }

  fun clearError() {
    mutable.update { it.copy(error = null) }
  }

  private fun entry(id: String): ModelEntry = state.value.tasks.first { it.taskId == id }

  /** Every initial, resumed and retried download requires a fresh explicit confirmation. */
  fun download(taskId: String) {
    if (downloads[taskId]?.isActive == true || taskId in deleting) return
    val entry = entry(taskId)
    if (!entry.canDownload) return
    try {
      ensureDownloadSpace(entry)
      mutable.update {
        it.copy(
          downloadConfirmation =
            DownloadConfirmation(taskId, entry.model, entry.totalBytes, isMetered())
        )
      }
    } catch (failure: Throwable) {
      showError(TaskFailures.message(failure))
    }
  }

  fun cancelDownloadConfirmation() {
    mutable.update { it.copy(downloadConfirmation = null) }
  }

  private fun isMetered(): Boolean =
    getApplication<Application>()
      .getSystemService(ConnectivityManager::class.java)
      ?.isActiveNetworkMetered ?: true

  private fun ensureDownloadSpace(model: ModelEntry) {
    val available = getApplication<Application>().filesDir.usableSpace
    val reserved =
      downloads
        .filterValues { it.isActive }
        .keys
        .filter { it != model.taskId }
        .sumOf { DownloadSafety.requiredFreeBytes(entry(it).totalBytes) }
    check(DownloadSafety.hasSpace(model.totalBytes, available, reserved)) {
      "Not enough storage. This download needs ${DownloadSafety.requiredFreeBytes(model.totalBytes)} free bytes " +
        "(twice its size), in addition to space reserved for other downloads."
    }
  }

  fun confirmDownload() {
    val confirmation = state.value.downloadConfirmation ?: return
    val taskId = confirmation.taskId
    if (downloads[taskId]?.isActive == true || taskId in deleting) return
    val entry = entry(taskId)
    try {
      ensureDownloadSpace(entry)
      if (isMetered() && !confirmation.metered) {
        mutable.update { it.copy(downloadConfirmation = confirmation.copy(metered = true)) }
        return
      }
    } catch (failure: Throwable) {
      cancelDownloadConfirmation()
      showError(TaskFailures.message(failure))
      return
    }
    cancelDownloadConfirmation()
    // Publish before the worker verifies cached files or waits for a network response.
    mutable.update { ui ->
      val previous = ui.downloads[taskId] ?: DownloadState(totalBytes = entry.totalBytes)
      ui.copy(
        downloads =
          ui.downloads + (taskId to previous.copy(status = DownloadStatus.STARTING, error = null))
      )
    }
    downloads[taskId] =
      viewModelScope.launch {
        try {
          ensureDownloadSpace(entry)
          store.download(entry) { progress ->
            mutable.update { it.copy(downloads = it.downloads + (taskId to progress)) }
          }
        } catch (e: CancellationException) {
          mutable.update { ui ->
            val old = ui.downloads[taskId] ?: DownloadState(totalBytes = entry.totalBytes)
            ui.copy(downloads = ui.downloads + (taskId to old.copy(status = DownloadStatus.PAUSED)))
          }
          throw e
        } catch (e: Throwable) {
          Log.e("ModelZooDownloads", "Model download failed: $taskId", e)
          mutable.update { ui ->
            val old = ui.downloads[taskId] ?: DownloadState(totalBytes = entry.totalBytes)
            ui.copy(
              downloads =
                ui.downloads +
                  (taskId to
                    old.copy(status = DownloadStatus.ERROR, error = TaskFailures.message(e)))
            )
          }
        } finally {
          withContext(NonCancellable) { refreshStorage() }
        }
      }
  }

  fun pauseDownload(taskId: String) {
    downloads[taskId]?.cancel()
  }

  fun delete(taskId: String) {
    if (state.value.busy || state.value.camera || state.value.recording || !deleting.add(taskId))
      return
    mutable.update { it.copy(busy = true) }
    viewModelScope.launch {
      downloads[taskId]?.cancel()
      downloads[taskId]?.join()
      try {
        withContext(inferenceDispatcher) {
          if (taskId == "object-detection") {
            detector?.close()
            detector = null
          }
          if (taskId == "text-to-speech") {
            synthesizer?.close()
            synthesizer = null
            speech = null
            speechCached = false
          }
          if (taskId == "speech-recognition") {
            recognizer?.close()
            recognizer = null
          }
          if (singleImageEngine?.first == taskId) {
            singleImageEngine?.second?.close()
            singleImageEngine = null
          }
          if (batchAudioEngine?.first == taskId) {
            batchAudioEngine?.second?.close()
            batchAudioEngine = null
            audioWaveforms = emptyList()
          }
        }
        store.delete(entry(taskId))
        mutable.update {
          it.copy(
            downloads = it.downloads + (taskId to DownloadState()),
            audioReady = if (taskId == "text-to-speech") false else it.audioReady,
            speechExportReady = if (taskId == "text-to-speech") false else it.speechExportReady,
            speechSaved = if (taskId == "text-to-speech") false else it.speechSaved,
          )
        }
        refreshStorage()
      } catch (e: Throwable) {
        showError(TaskFailures.message(e))
      } finally {
        deleting.remove(taskId)
        mutable.update { it.copy(busy = false) }
      }
    }
  }

  private suspend fun refreshStorage() {
    val bytes = store.storageBytes()
    mutable.update { it.copy(storageBytes = bytes) }
  }

  private fun detectionEngine(): DetectionEngine {
    val entry = entry("object-detection")
    check(state.value.downloads[entry.taskId]?.status == DownloadStatus.READY) {
      "Download and verify the detection model first"
    }
    return detector ?: DetectionEngine(store.directory(entry), entry.backend).also { detector = it }
  }

  fun pickImage(uri: Uri) {
    val taskId = state.value.selectedTaskId
    val generation = navigationGeneration
    if (taskId in SingleImageTasks.ids) {
      runTask {
        val bitmap = withContext(Dispatchers.IO) { readImage(uri) }
        publishImageInput(bitmap, taskId!!, generation)
      }
      return
    }
    if (taskId != "object-detection") return
    runTask {
      val bitmap = withContext(Dispatchers.IO) { readImage(uri) }
      publishImageInput(bitmap, taskId, generation)
    }
  }

  fun photograph(bitmap: Bitmap) {
    val taskId = state.value.selectedTaskId
    val generation = navigationGeneration
    if (state.value.busy || state.value.recording) {
      bitmap.recycle()
      return
    }
    if (taskId in SingleImageTasks.ids) {
      publishImageInput(bitmap, taskId!!, generation)
      return
    }
    if (taskId != "object-detection") {
      bitmap.recycle()
      return
    }
    publishImageInput(bitmap, taskId, generation)
  }

  /** A chosen photo remains an input until the user explicitly taps Run. */
  fun runDetectionImage() {
    val snapshot = state.value
    val bitmap = snapshot.inputImage ?: return
    if (snapshot.selectedTaskId != "object-detection" || snapshot.busy) return
    runTask { publishDetection(bitmap, realtime = false) }
  }

  fun pickSecondImage(uri: Uri) {
    val generation = navigationGeneration
    if (state.value.selectedTaskId != "image-matching") return
    runTask {
      val bitmap = withContext(Dispatchers.IO) { readImage(uri) }
      if (generation != navigationGeneration || cleared) bitmap.recycle()
      else
        mutable.update {
          it.copy(
            secondaryImage = bitmap,
            outputImage = null,
            imageOutputText = "",
            imageOutputDetails = "",
          )
        }
    }
  }

  private fun publishImageInput(bitmap: Bitmap, taskId: String, generation: Long) {
    if (cleared || generation != navigationGeneration || state.value.selectedTaskId != taskId) {
      bitmap.recycle()
      return
    }
    val displayed =
      if (taskId == "super-resolution-real-esrgan" && maxOf(bitmap.width, bitmap.height) > 512) {
        val scale = 512f / maxOf(bitmap.width, bitmap.height)
        Bitmap.createScaledBitmap(
            bitmap,
            maxOf(1, (bitmap.width * scale).toInt()),
            maxOf(1, (bitmap.height * scale).toInt()),
            true,
          )
          .also { if (it !== bitmap) bitmap.recycle() }
      } else bitmap
    mutable.update {
      if (cleared || generation != navigationGeneration || it.selectedTaskId != taskId) it
      else
        it.copy(
          inputImage = displayed,
          image = null,
          boxes = emptyList(),
          outputImage = null,
          imageOutputText = "",
          imageOutputDetails = "",
          inferenceMs = null,
          backend = "",
          fallbackReason = null,
          backendDetails = "",
          error = null,
        )
    }
  }

  fun runSingleImage() {
    val ui = state.value
    val taskId = ui.selectedTaskId
    val image = ui.inputImage
    if (ui.busy || image == null || taskId !in SingleImageTasks.ids) return
    val generation = navigationGeneration
    mutable.update {
      it.copy(
        outputImage = null,
        imageOutputText = "",
        imageOutputDetails = "",
        inferenceMs = null,
        fallbackReason = null,
        backendDetails = "",
      )
    }
    runTask {
      var input: Bitmap? = null
      var secondary: Bitmap? = null
      try {
        val entry = entry(taskId!!)
        check(ui.downloads[taskId]?.status == DownloadStatus.READY) {
          "Download and verify the model first"
        }
        if (cleared || generation != navigationGeneration) return@runTask
        if (singleImageEngine?.first != taskId) {
          singleImageEngine?.second?.close()
          singleImageEngine = null
          singleImageEngine =
            taskId to
              SingleImageTasks.create(
                taskId,
                getApplication(),
                store.directory(entry),
                entry.backend,
              )
        }
        // Protect the displayed source image from source-wrapper bitmap ownership conventions.
        val working = image.copy(Bitmap.Config.ARGB_8888, false)
        input = working
        secondary = ui.secondaryImage?.copy(Bitmap.Config.ARGB_8888, false)
        val result =
          singleImageEngine!!
            .second
            .run(ImageTaskRequest(working, secondaryBitmap = secondary))
        val output =
          if (result.bitmap === working) working.copy(Bitmap.Config.ARGB_8888, false)
          else result.bitmap
        if (cleared || generation != navigationGeneration || state.value.selectedTaskId != taskId) {
          output?.recycle()
          return@runTask
        }
        mutable.update {
          if (cleared || generation != navigationGeneration || it.selectedTaskId != taskId) it
          else
            it.copy(
              outputImage = output,
              imageOutputText = result.text,
              imageOutputDetails = result.details,
              inferenceMs = result.inferenceMs,
              backend = result.backend,
              fallbackReason = result.fallbackReason,
              backendDetails = result.backendDetails,
            )
        }
      } finally {
        input?.takeUnless { it.isRecycled }?.recycle()
        secondary?.takeUnless { it.isRecycled }?.recycle()
      }
    }
  }

  fun setCamera(enabled: Boolean) {
    if (enabled && state.value.busy) return
    cameraEnabled = enabled
    mutable.update {
      it.copy(
        camera = enabled,
        cameraFrames = if (enabled) 0 else it.cameraFrames,
        error = if (enabled) null else it.error,
      )
    }
  }

  /** The canonical camera pipeline owns this bitmap until this synchronous callback returns. */
  fun onCameraFrame(bitmap: Bitmap) {
    if (!cameraEnabled || cleared) return
    runBlocking(inferenceDispatcher) {
      if (!cameraEnabled || cleared) return@runBlocking
      try {
        if (state.value.selectedTaskId in RealtimeImageTasks.ids) publishRealtimeImage(bitmap)
        else publishDetection(bitmap, realtime = true)
      } catch (e: Throwable) {
        cameraEnabled = false
        mutable.update { it.copy(camera = false, error = TaskFailures.message(e)) }
      }
    }
  }

  private fun publishRealtimeImage(bitmap: Bitmap) {
    val taskId = state.value.selectedTaskId ?: return
    val generation = navigationGeneration
    val entry = entry(taskId)
    check(state.value.downloads[taskId]?.status == DownloadStatus.READY) {
      "Download and verify the model first"
    }
    if (singleImageEngine?.first != taskId) {
      singleImageEngine?.second?.close()
      singleImageEngine =
        taskId to
          SingleImageTasks.create(taskId, getApplication(), store.directory(entry), entry.backend)
    }
    // The camera pool retains its bitmap; each wrapper receives an owned disposable copy.
    val working = bitmap.copy(Bitmap.Config.ARGB_8888, false)
    try {
      val result = singleImageEngine!!.second.run(ImageTaskRequest(working))
      val display =
        result.bitmap?.let { if (it === working) it.copy(Bitmap.Config.ARGB_8888, false) else it }
          ?: bitmap.copy(Bitmap.Config.ARGB_8888, false)
      if (generation != navigationGeneration || !cameraEnabled || cleared) {
        display.recycle()
        return
      }
      mutable.update {
        if (generation != navigationGeneration || !cameraEnabled || it.selectedTaskId != taskId) it
        else
          it.copy(
            outputImage = display,
            imageOutputText = result.text,
            imageOutputDetails = result.details,
            inferenceMs = result.inferenceMs,
            backend = result.backend,
            fallbackReason = result.fallbackReason,
            backendDetails = result.backendDetails,
            cameraFrames = it.cameraFrames + 1,
          )
      }
    } finally {
      if (!working.isRecycled) working.recycle()
    }
  }

  private fun publishDetection(bitmap: Bitmap, realtime: Boolean) {
    val generation = navigationGeneration
    val engine = detectionEngine()
    val result = engine.detect(bitmap)
    if (cleared || state.value.selectedTaskId != "object-detection" || (realtime && !cameraEnabled))
      return
    val snapshot = bitmap.copy(Bitmap.Config.ARGB_8888, false)
    mutable.update {
      if (
        cleared ||
          generation != navigationGeneration ||
          it.selectedTaskId != "object-detection" ||
          (realtime && !cameraEnabled)
      )
        return@update it
      it.copy(
        image = snapshot,
        boxes = result.boxes,
        labels = engine.labels,
        inferenceMs = result.inferenceMs.toDouble(),
        backend = result.backend,
        fallbackReason = result.fallbackReason,
        cameraFrames = if (realtime) it.cameraFrames + 1 else it.cameraFrames,
      )
    }
  }

  fun synthesize() {
    val text = state.value.inputText
    val generation = navigationGeneration
    if (state.value.playing) stopPlayback()
    runTask {
      val entry = entry("text-to-speech")
      check(state.value.downloads[entry.taskId]?.status == DownloadStatus.READY) {
        "Download and verify the speech model first"
      }
      val engine =
        synthesizer ?: MatchaEngine(store.directory(entry), entry.backend).also { synthesizer = it }
      val result = engine.synthesize(text)
      if (
        cleared ||
          generation != navigationGeneration ||
          state.value.selectedTaskId != "text-to-speech"
      )
        return@runTask
      speech = result.samples
      speechRate = result.sampleRate
      speechCached = false
      try {
        withContext(Dispatchers.IO) { speechOutput.save(result.samples, result.sampleRate) }
        speechCached = true
      } catch (failure: CancellationException) {
        throw failure
      } catch (failure: Throwable) {
        showError(
          getApplication<Application>()
            .getString(R.string.speech_cache_failed, TaskFailures.message(failure))
        )
      }
      mutable.update {
        if (cleared || generation != navigationGeneration || it.selectedTaskId != "text-to-speech")
          return@update it
        it.copy(
          audioReady = true,
          speechExportReady = speechCached,
          speechSaved = false,
          playbackElapsedSeconds = 0f,
          playbackTotalSeconds = result.samples.size / result.sampleRate.toFloat(),
          inferenceMs = result.inferenceMs,
          backend = result.backend,
          fallbackReason = result.fallbackReason,
          backendDetails = result.backendDetails,
        )
      }
    }
  }

  fun startRecording() {
    if (
      state.value.busy ||
        state.value.recording ||
        state.value.selectedTaskId !in (BatchAudioTasks.ids + "speech-recognition")
    )
      return
    val selectedTask = state.value.selectedTaskId ?: return
    if (state.value.downloads[selectedTask]?.status != DownloadStatus.READY) {
      showError("Download and verify the audio model first")
      return
    }
    val session =
      try {
        RecordingSession(navigationGeneration, selectedTask)
      } catch (failure: Throwable) {
        showError(TaskFailures.message(failure))
        return
      }
    synchronized(captureLock) { recordingSession = session }
    mutable.update {
      it.copy(
        recording = true,
        recordedSeconds = 0f,
        audioInputName = null,
        audioInputSeconds = null,
        transcript = null,
        inferenceMs = null,
        error = null,
      )
    }
    viewModelScope.launch(inferenceDispatcher) {
      try {
        synchronized(captureLock) {
          if (recordingSession !== session) return@launch
          session.capture.start onChunk@{ chunk ->
            val count: Int
            synchronized(captureLock) {
              if (recordingSession !== session) return@onChunk
              val copied = minOf(chunk.size, session.samples.size - session.count)
              System.arraycopy(chunk, 0, session.samples, session.count, copied)
              session.count += copied
              count = session.count
            }
            mutable.update {
              if (it.recording && session.generation == navigationGeneration)
                it.copy(recordedSeconds = count / 16000f)
              else it
            }
            if (count == session.samples.size) {
              viewModelScope.launch {
                if (synchronized(captureLock) { recordingSession === session }) stopRecording()
              }
            }
          }
        }
      } catch (e: Throwable) {
        if (e is CancellationException) throw e
        viewModelScope.launch {
          if (synchronized(captureLock) { recordingSession === session }) {
            stopRecording(transcribe = false)
            showError(TaskFailures.message(e))
          }
        }
      }
    }
  }

  fun stopRecording(transcribe: Boolean = true) {
    val session =
      synchronized(captureLock) {
        val current = recordingSession ?: return
        recordingSession = null
        current.capture.stop()
        current
      }
    mutable.update { it.copy(recording = false) }
    if (
      transcribe &&
        session.generation == navigationGeneration &&
        state.value.selectedTaskId == session.taskId
    ) {
      val samples = session.samples.copyOf(session.count)
      mutable.update {
        it.copy(
          audioInputName = getApplication<Application>().getString(R.string.microphone_recording),
          audioInputSeconds = samples.size / 16000f,
        )
      }
      runTask {
        if (session.taskId in BatchAudioTasks.ids)
          processBatchAudio(samples, 16000, session.taskId, session.generation)
        else transcribeSamples(samples, session.generation)
      }
    }
  }

  fun pickWav(uri: Uri) {
    val generation = navigationGeneration
    val taskId = state.value.selectedTaskId ?: return
    runTask {
      val (samples, displayName) =
        withContext(Dispatchers.IO) {
          val bytes =
            getApplication<Application>().contentResolver.openInputStream(uri).use { stream ->
              requireNotNull(stream) { "Could not open the WAV file" }
              val output = ByteArrayOutputStream()
              val buffer = ByteArray(64 * 1024)
              while (true) {
                val n = stream.read(buffer)
                if (n < 0) break
                require(output.size().toLong() + n <= 32L * 1024 * 1024) {
                  "Choose a WAV file smaller than 32 MB"
                }
                output.write(buffer, 0, n)
              }
              output.toByteArray()
            }
          WavAudio.readMono(bytes) to audioDisplayName(uri)
        }
      if (cleared || generation != navigationGeneration || state.value.selectedTaskId != taskId)
        return@runTask
      mutable.update {
        it.copy(
          audioInputName = displayName,
          audioInputSeconds = samples.samples.size / samples.sampleRate.toFloat(),
        )
      }
      if (taskId in BatchAudioTasks.ids)
        processBatchAudio(samples.samples, samples.sampleRate, taskId, generation)
      else
        transcribeSamples(WavAudio.resample(samples.samples, samples.sampleRate, 16000), generation)
    }
  }

  private fun processBatchAudio(
    samples: FloatArray,
    sampleRate: Int,
    taskId: String,
    generation: Long,
  ) {
    if (cleared || generation != navigationGeneration) return
    val entry = entry(taskId)
    check(state.value.downloads[taskId]?.status == DownloadStatus.READY) {
      "Download and verify the model first"
    }
    if (batchAudioEngine?.first != taskId) {
      batchAudioEngine?.second?.close()
      batchAudioEngine =
        taskId to
          BatchAudioTasks.create(taskId, store.directory(entry), entry.backend)
    }
    val engine = batchAudioEngine!!.second
    mutable.update { it.copy(audioProgress = null) }
    val result =
      engine.run(
        AudioTaskRequest(samples, sampleRate) { stem, chunk, total ->
          mutable.update {
            if (cleared || generation != navigationGeneration || it.selectedTaskId != taskId) it
            else it.copy(audioProgress = AudioStageProgress(stem, chunk, total))
          }
        }
      )
    if (cleared || generation != navigationGeneration || state.value.selectedTaskId != taskId)
      return
    audioWaveforms = listOf(NamedWaveform("Input", samples, sampleRate)) + result.waveforms
    val pitch =
      (result.metrics["f0Hz"] as? List<*>)?.mapNotNull { (it as? Number)?.toFloat() }.orEmpty()
    val confidence =
      (result.metrics["confidence"] as? List<*>)
        ?.mapNotNull { (it as? Number)?.toFloat() }
        .orEmpty()
    mutable.update {
      if (generation != navigationGeneration || it.selectedTaskId != taskId) it
      else
        it.copy(
          audioSummary = result.summary,
          audioOutputs = audioWaveforms.map { wave -> wave.name },
          inferenceMs = result.inferenceMs,
          backend = engine.backend.actual,
          fallbackReason = engine.backend.fallbackReason,
          backendDetails = engine.backend.details,
          pitchHz = pitch,
          pitchConfidence = confidence,
          pitchHopSeconds = (result.metrics["hopSeconds"] as? Number)?.toFloat() ?: 0.1f,
        )
    }
  }

  fun playAudioOutput(index: Int) {
    val wave = audioWaveforms.getOrNull(index) ?: return
    playSamples(wave.samples, wave.sampleRate)
  }

  private fun transcribeSamples(samples: FloatArray, generation: Long) {
    if (cleared || generation != navigationGeneration) return
    val entry = entry("speech-recognition")
    check(state.value.downloads[entry.taskId]?.status == DownloadStatus.READY) {
      "Download and verify Speech Recognition first"
    }
    val engine =
      recognizer
        ?: ZipformerEngine(getApplication(), store.directory(entry), entry.backend).also {
          recognizer = it
        }
    val result = engine.transcribe(samples)
    mutable.update {
      if (
        cleared || generation != navigationGeneration || it.selectedTaskId != "speech-recognition"
      )
        return@update it
      it.copy(
        transcript = result.text,
        inferenceMs = result.inferenceMs,
        backend = result.backend,
        fallbackReason = result.fallbackReason,
        backendDetails = "",
      )
    }
  }

  fun playSpeech() {
    val samples = speech ?: return
    playSamples(samples, speechRate)
  }

  private fun playSamples(samples: FloatArray, sampleRate: Int) {
    if (state.value.playing) return
    val generation = navigationGeneration
    mutable.update {
      it.copy(
        playing = true,
        playbackElapsedSeconds = 0f,
        playbackTotalSeconds = samples.size / sampleRate.toFloat(),
      )
    }
    val ticket = player.reserve()
    viewModelScope.launch(Dispatchers.IO) {
      try {
        player.play(samples, sampleRate, ticket) { elapsed, total ->
          mutable.update {
            if (generation != navigationGeneration || cleared) it
            else it.copy(playbackElapsedSeconds = elapsed, playbackTotalSeconds = total)
          }
        }
      } catch (e: Throwable) {
        if (generation == navigationGeneration && player.isCurrent(ticket))
          showError(TaskFailures.message(e))
      } finally {
        if (generation == navigationGeneration && player.isCurrent(ticket))
          mutable.update { it.copy(playing = false) }
      }
    }
  }

  /** Only called after the user chooses a destination in Android's Save dialog. */
  fun saveSpeech(uri: Uri) {
    val snapshot = state.value
    if (!snapshot.speechExportReady || snapshot.busy || snapshot.speechSaving) return
    mutable.update { it.copy(speechSaving = true, speechSaved = false) }
    viewModelScope.launch(Dispatchers.IO) {
      try {
        val resolver = getApplication<Application>().contentResolver
        checkNotNull(resolver.openOutputStream(uri, "w")) { "The chosen file is unavailable." }
          .use { speechOutput.copyTo(it) }
        mutable.update { it.copy(speechSaved = it.selectedTaskId == "text-to-speech") }
      } catch (failure: CancellationException) {
        throw failure
      } catch (failure: Throwable) {
        showError(
          getApplication<Application>()
            .getString(R.string.speech_export_failed, TaskFailures.message(failure))
        )
      } finally {
        mutable.update { it.copy(speechSaving = false) }
      }
    }
  }

  fun stopPlayback() {
    player.close()
    mutable.update { it.copy(playing = false) }
  }

  private fun runTask(block: suspend () -> Unit) {
    if (state.value.busy || state.value.recording) return
    mutable.update { it.copy(busy = true, error = null) }
    viewModelScope.launch(inferenceDispatcher) {
      try {
        block()
      } catch (e: CancellationException) {
        throw e
      } catch (e: Throwable) {
        showError(TaskFailures.message(e))
      } finally {
        mutable.update { it.copy(busy = false) }
      }
    }
  }

  private fun readImage(uri: Uri): Bitmap {
    val resolver = getApplication<Application>().contentResolver
    val dimensions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    resolver.openInputStream(uri).use {
      checkNotNull(it)
      BitmapFactory.decodeStream(it, null, dimensions)
    }
    require(dimensions.outWidth > 0 && dimensions.outHeight > 0) { "Unsupported image" }
    // The preview scales only while drawing. Inference receives the original decoded pixels,
    // including OCR documents larger than 2048 pixels; no JPEG recompression is introduced.
    val bitmap =
      resolver.openInputStream(uri).use {
        checkNotNull(it)
        requireNotNull(
          BitmapFactory.decodeStream(
            it,
            null,
            BitmapFactory.Options().apply {
              inSampleSize = 1
              inScaled = false
              inPreferredConfig = Bitmap.Config.ARGB_8888
            },
          )
        ) {
          "Could not read image"
        }
      }
    val orientation =
      resolver.openInputStream(uri).use { input ->
        if (input == null) ExifInterface.ORIENTATION_NORMAL
        else
          runCatching {
              ExifInterface(input)
                .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            }
            .getOrDefault(ExifInterface.ORIENTATION_NORMAL)
      }
    val matrix = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> {
        matrix.setRotate(90f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> {
        matrix.setRotate(270f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(270f)
    }
    if (matrix.isIdentity) return bitmap
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true).also {
      if (it !== bitmap) bitmap.recycle()
    }
  }

  private fun audioDisplayName(uri: Uri): String {
    val fromProvider =
      runCatching {
          getApplication<Application>()
            .contentResolver
            .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
        }
        .getOrNull()
    return fromProvider?.takeIf { it.isNotBlank() }
      ?: uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() }
      ?: "Audio.wav"
  }

  override fun onCleared() {
    cleared = true
    cameraEnabled = false
    stopRecording(transcribe = false)
    player.close()
    cleanupScope.launch {
      try {
        detector?.close()
        synthesizer?.close()
        recognizer?.close()
        singleImageEngine?.second?.close()
        batchAudioEngine?.second?.close()
      } catch (failure: Throwable) {
        if (failure is CancellationException) throw failure
        Log.e("ModelZooTask", "Model cleanup failed", failure)
      } finally {
        cleanupScope.cancel()
        inferenceDispatcher.close()
      }
    }
    super.onCleared()
  }

  companion object {
    fun factory(application: Application) =
      object : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
          require(modelClass.isAssignableFrom(MainViewModel::class.java))
          @Suppress("UNCHECKED_CAST")
          return MainViewModel(application) as T
        }
      }
  }
}
