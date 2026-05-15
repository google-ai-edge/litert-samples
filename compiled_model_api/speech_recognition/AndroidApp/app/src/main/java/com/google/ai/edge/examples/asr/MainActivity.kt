/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.asr

import android.content.ClipData
import android.content.ClipboardManager
import android.graphics.Color
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.style.ForegroundColorSpan
import android.view.View
import android.widget.ArrayAdapter
import android.widget.ScrollView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.ai.edge.examples.asr.databinding.ActivityMainBinding
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.NpuCompatibilityChecker
import java.util.Locale
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield

class MainActivity : AppCompatActivity() {
  private lateinit var binding: ActivityMainBinding
  private lateinit var modelMetadataManager: ModelMetadataManager

  private var selectedAudioUri: Uri? = null

  // Note that these 2 objects are accessed on the IO thread only.
  private var audioPreprocessor: AudioPreprocessor? = null
  private var speechRecognizer: SpeechRecognizer? = null
  private var tokenizer: Tokenizer? = null
  private var modelInputInterval: Duration = 0.seconds

  private var mediaPlayer: MediaPlayer? = null
  private var playbackJob: Job? = null
  private var loadJob: Job? = null

  private var recordingJob: Job? = null
  private val isRecording
    get() = recordingJob != null

  private val requestPermissionLauncher =
    registerForActivityResult(
      androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { isGranted ->
      if (isGranted) {
        toggleRecording()
      } else {
        Toast.makeText(this, "Microphone permission denied", Toast.LENGTH_SHORT).show()
      }
    }

  private val pickAudioLauncher =
    registerForActivityResult(
      androidx.activity.result.contract.ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
      if (uri != null) {
        selectedAudioUri = uri
        val fileName = getFileName(uri)
        binding.contentMain.selectedAudioName.text = fileName
        prepareMediaPlayer(uri)
        Toast.makeText(this, "Audio selected: $fileName", Toast.LENGTH_SHORT).show()
      } else {
        Toast.makeText(this, "No audio selected", Toast.LENGTH_SHORT).show()
      }
    }

  private fun hasRecordAudioPermission() =
    androidx.core.content.ContextCompat.checkSelfPermission(
      this,
      android.Manifest.permission.RECORD_AUDIO,
    ) == android.content.pm.PackageManager.PERMISSION_GRANTED

  private fun getFileName(uri: Uri): String {
    contentResolver.query(uri, null, null, null, null)?.use { cursor ->
      val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
      if (nameIndex != -1 && cursor.moveToFirst()) {
        return@getFileName cursor.getString(nameIndex)
      }
    }
    return uri.lastPathSegment ?: "Selected Audio"
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)

    // Disable buttons until model is ready
    binding.contentMain.transcribeButton.isEnabled = false

    binding.contentMain.selectAudioButton.setOnClickListener { openAudioFilePicker() }
    binding.contentMain.recordAudioButton.setOnClickListener {
      if (hasRecordAudioPermission()) {
        toggleRecording()
      } else {
        requestPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
      }
    }
    binding.contentMain.transcribeButton.setOnClickListener { startTranscription() }
    binding.contentMain.copyButton.setOnClickListener { copyToClipboard() }
    binding.contentMain.clearButton.setOnClickListener { clearTranscription() }

    val metadataJson = assets.open(MODEL_METADATA_PATH).bufferedReader().use { it.readText() }
    modelMetadataManager = ModelMetadataManager(metadataJson)

    val modelOptions = modelMetadataManager.getAvailableModels()
    val modelAdapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, modelOptions)
    binding.contentMain.modelSelector.setAdapter(modelAdapter)
    binding.contentMain.modelSelector.setOnItemClickListener { _, _, _, _ -> preparePipeline() }

    val npuString = getString(if (hasTpu()) R.string.tpu else R.string.npu)
    val backendOptions = listOf(getString(R.string.cpu), getString(R.string.gpu), npuString)
    val backendAdapter =
      ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, backendOptions)
    binding.contentMain.backendSelector.setAdapter(backendAdapter)
    binding.contentMain.backendSelector.setOnItemClickListener { _, _, _, _ -> preparePipeline() }

    setupPlayerListeners()
  }

  private fun getAcceleratorSelected(accelerator: String): Accelerator {
    when (accelerator) {
      getString(R.string.gpu) -> return Accelerator.GPU
      getString(R.string.npu) -> return Accelerator.NPU
      getString(R.string.tpu) -> return Accelerator.NPU
      else -> return Accelerator.CPU
    }
  }

  private fun setupPlayerListeners() {
    binding.contentMain.playPauseButton.setOnClickListener { togglePlayback() }

    binding.contentMain.audioSlider.addOnChangeListener { _, value, fromUser ->
      if (fromUser) {
        mediaPlayer?.seekTo(value.toInt())
        binding.contentMain.currentTime.text = formatTime(value.toInt())
      }
    }
  }

  private fun prepareMediaPlayer(uri: Uri) {
    mediaPlayer?.release()
    mediaPlayer =
      MediaPlayer().apply {
        setDataSource(applicationContext, uri)
        prepare()
        binding.contentMain.totalTime.text = formatTime(duration)
        binding.contentMain.audioSlider.valueTo = duration.toFloat()
        binding.contentMain.audioSlider.value = 0f
        binding.contentMain.playerLayout.visibility = View.VISIBLE
      }
    binding.contentMain.playPauseButton.setImageResource(android.R.drawable.ic_media_play)
  }

  private fun togglePlayback() {
    val player = mediaPlayer ?: return
    if (player.isPlaying) {
      player.pause()
      binding.contentMain.playPauseButton.setImageResource(android.R.drawable.ic_media_play)
      stopPlaybackUpdates()
    } else {
      player.start()
      binding.contentMain.playPauseButton.setImageResource(android.R.drawable.ic_media_pause)
      startPlaybackUpdates()
    }
  }

  private fun startPlaybackUpdates() {
    playbackJob?.cancel()
    playbackJob = lifecycleScope.launch {
      // isActive ensures the loop stops immediately if the job is cancelled
      while (isActive) {
        val player = mediaPlayer ?: break // Exit loop if player is released
        if (player.isPlaying) {
          binding.contentMain.audioSlider.value = player.currentPosition.toFloat()
          binding.contentMain.currentTime.text = formatTime(player.currentPosition)
        }
        delay(100)
      }
    }
  }

  private fun stopPlaybackUpdates() {
    playbackJob?.cancel()
  }

  private fun formatTime(ms: Int): String {
    val minutes = (ms / 1000) / 60
    val seconds = (ms / 1000) % 60
    return String.format(Locale.ENGLISH, "%02d:%02d", minutes, seconds)
  }

  private fun resetPipeline() {
    audioPreprocessor?.close()
    audioPreprocessor = null
    speechRecognizer?.close()
    speechRecognizer = null
    tokenizer?.close()
    tokenizer = null
  }

  private fun preparePipeline() {
    setModelLoadingState(true)

    // Prevent multiple concurrent initializations
    loadJob?.cancel()

    val modelKey = binding.contentMain.modelSelector.text.toString()
    val accelerator = getAcceleratorSelected(binding.contentMain.backendSelector.text.toString())

    // Load the model, heavy resources, on the IO thread.
    loadJob =
      lifecycleScope.launch(Dispatchers.IO) {
        resetPipeline()

        try {
          val modelConfig = modelMetadataManager.getModelConfig(modelKey)
          audioPreprocessor =
            if (modelConfig.logMelSpectro != null) {
              MelSpectroProcessor(SAMPLING_RATE, modelConfig.logMelSpectro)
            } else {
              DummyAudioProcessor()
            }
          tokenizer = HuggingfaceTokenizer(applicationContext, modelConfig)
          val decoderFactory: (CompiledModel, ModelConfig) -> LiteRtRunner.Decoder =
            when {
              modelKey.contains("-ctc-") -> { _, _ ->
                CtcDecoder(tokenizer!!.vocabSize)
              }
              modelKey.contains("-tdt-") -> { model, config ->
                TdtDecoder(model, config)
              }
              else -> { model, config ->
                LiteRtRunner.DefaultDecoder(model, config)
              }
            }
          speechRecognizer =
            LiteRtRunner(this@MainActivity, modelConfig, accelerator, decoderFactory)
          modelInputInterval = modelConfig.inputMilliseconds.milliseconds

          // Update UI on the main thread
          withContext(Dispatchers.Main) {
            setModelLoadingState(false)
            Toast.makeText(this@MainActivity, "$modelKey ready", Toast.LENGTH_SHORT).show()
          }
        } catch (e: CancellationException) {
          resetPipeline()
          throw e
        } catch (e: Exception) {
          resetPipeline()
          withContext(Dispatchers.Main) {
            setModelLoadingState(false)
            val errorMessage = getString(R.string.error_loading_model, e.message)
            binding.contentMain.transcribeButton.text = errorMessage
            Toast.makeText(this@MainActivity, errorMessage, Toast.LENGTH_LONG).show()
          }
          e.printStackTrace()
        }
      }
  }

  private fun setModelLoadingState(isLoading: Boolean) {
    val transcribeButton = binding.contentMain.transcribeButton
    transcribeButton.isEnabled = !isLoading
    if (isLoading) {
      val modelKey = binding.contentMain.modelSelector.text.toString()
      setLongOperationState(inProgress = true, label = getString(R.string.loading_model, modelKey))
    } else {
      transcribeButton.text = getString(R.string.transcribe)
      setLongOperationState(inProgress = false)
    }
  }

  private fun openAudioFilePicker() {
    try {
      pickAudioLauncher.launch("audio/*")
    } catch (e: Exception) {
      Toast.makeText(this, e.localizedMessage, Toast.LENGTH_SHORT).show()
    }
  }

  private fun startTranscription() {
    val uri = selectedAudioUri
    if (uri == null) {
      Toast.makeText(this, "Please select an audio file first.", Toast.LENGTH_SHORT).show()
      return
    }

    binding.contentMain.transcriptionText.text = ""
    setLongOperationState(inProgress = true, label = getString(R.string.transcribing))

    lifecycleScope.launch(Dispatchers.IO) {
      if (speechRecognizer == null) {
        withContext(Dispatchers.Main) {
          Toast.makeText(this@MainActivity, "Model not ready yet.", Toast.LENGTH_SHORT).show()
          setLongOperationState(inProgress = false)
        }
        return@launch // Exit the coroutine early
      }

      processFileAudio(uri)

      withContext(Dispatchers.Main) { setLongOperationState(inProgress = false) }
    }
  }

  private fun setLongOperationState(inProgress: Boolean, label: String = "") {
    val visibility = if (inProgress) View.VISIBLE else View.GONE
    binding.contentMain.longOperationProgress.visibility = visibility
    binding.contentMain.longOperationLabel.text = label
    binding.contentMain.longOperationLabel.visibility = visibility
    binding.contentMain.transcribeButton.isEnabled = !inProgress
    binding.contentMain.selectAudioButton.isEnabled = !inProgress
    binding.contentMain.recordAudioButton.isEnabled = !inProgress || isRecording
    if (inProgress) {
      binding.contentMain.playerLayout.visibility = View.GONE
    } else if (selectedAudioUri != null) {
      binding.contentMain.playerLayout.visibility = View.VISIBLE
    }
  }

  private fun toggleRecording() {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  private fun startRecording() {
    binding.contentMain.recordAudioButton.text = getString(R.string.stop_recording)
    binding.contentMain.recordAudioButton.setIconResource(android.R.drawable.ic_media_pause)
    binding.contentMain.transcriptionText.text = ""

    recordingJob =
      lifecycleScope.launch(Dispatchers.IO) {
        if (speechRecognizer == null || audioPreprocessor == null || tokenizer == null) {
          withContext(Dispatchers.Main) {
            Toast.makeText(this@MainActivity, "Model not ready yet.", Toast.LENGTH_SHORT).show()
            stopRecording()
          }
          return@launch
        }

        try {
          val audioInterval = LIVE_AUDIO_CHUNK_INTERVAL.coerceAtMost(modelInputInterval)
          val audioOverlap = audioInterval - LIVE_AUDIO_CHUNK_SLIDING_INTERVAL
          MicrophoneAudioSource(SAMPLING_RATE, audioInterval, audioOverlap).use { audioSource ->
            transcribeAudio(
              audioSource,
              (audioOverlap / audioInterval).toFloat(),
              LIVE_MAX_LEVENSHTEIN_DISTANCE,
            )
          }
        } catch (e: CancellationException) {
          // Handled silently
        } catch (e: Exception) {
          e.printStackTrace()
          withContext(Dispatchers.Main) {
            Toast.makeText(this@MainActivity, "Recording error: ${e.message}", Toast.LENGTH_LONG)
              .show()
            stopRecording()
          }
        }
      }

    setLongOperationState(inProgress = true, label = getString(R.string.transcribing))
  }

  private fun stopRecording() {
    recordingJob?.cancel()
    recordingJob = null
    binding.contentMain.recordAudioButton.text = getString(R.string.start_recording)
    binding.contentMain.recordAudioButton.setIconResource(android.R.drawable.ic_btn_speak_now)
    setLongOperationState(inProgress = false)

    // Remove the gray span from any remaining unconfirmed text to finalize it
    val currentText = binding.contentMain.transcriptionText.text.toString()
    binding.contentMain.transcriptionText.text = currentText
  }

  private suspend fun transcribeAudio(
    audioSource: AudioSource,
    overlapRatio: Float,
    maxLevenshteinDistance: Int,
  ) {
    val preprocessedChannel = Channel<FloatArray>(Channel.BUFFERED)
    val tokenChannel = Channel<Pair<Int, Int>>(Channel.BUFFERED)
    LevenshteinTokenMerger(
        tokenizer!!,
        overlapRatio,
        maxLevenshteinDistance = maxLevenshteinDistance,
      )
      .use { postprocessor ->
        coroutineScope {
          scheduleAudioPreprocessing(audioSource, audioPreprocessor!!, preprocessedChannel)
          scheduleSpeechRecognition(speechRecognizer!!, preprocessedChannel, tokenChannel)
          scheduleTokenPostprocessing(postprocessor, tokenChannel)
        }
      }
  }

  private fun CoroutineScope.scheduleAudioPreprocessing(
    audioSource: AudioSource,
    audioPreprocessor: AudioPreprocessor,
    preprocessedChannel: Channel<FloatArray>,
  ) =
    launch(Dispatchers.IO) {
      try {
        for (audioChunk in audioSource.getAudioData()) {
          val preprocessed = audioPreprocessor.process(audioChunk)
          preprocessedChannel.send(preprocessed)
          yield()
        }
      } finally {
        preprocessedChannel.close()
      }
    }

  private fun CoroutineScope.scheduleSpeechRecognition(
    speechRecognizer: SpeechRecognizer,
    preprocessedChannel: Channel<FloatArray>,
    tokenChannel: Channel<Pair<Int, Int>>,
  ) =
    launch(Dispatchers.IO) {
      try {
        for (preprocessed in preprocessedChannel) {
          val tokenSequence = speechRecognizer.recognize(preprocessed)
          for ((tokenId, timestamp) in tokenSequence) {
            tokenChannel.send(Pair(tokenId, timestamp))
            if (tokenId == SpeechRecognizer.END_OF_SEQUENCE) break
          }
          yield()
        }
      } finally {
        tokenChannel.close()
      }
    }

  private fun CoroutineScope.scheduleTokenPostprocessing(
    postprocessor: Postprocessor,
    tokenChannel: Channel<Pair<Int, Int>>,
  ) =
    launch(Dispatchers.Default) {
      var totalConfirmedText = ""
      for ((tokenId, timestamp) in tokenChannel) {
        val decodedText = postprocessor.decode(tokenId, timestamp)
        if (decodedText != null) {
          val (confirmedText, unconfirmedText) = decodedText
          totalConfirmedText = merge(totalConfirmedText, confirmedText)
          withContext(Dispatchers.Main) {
            updateTranscriptionUI(totalConfirmedText, unconfirmedText)
          }
        }
      }

      // Clear the color of unconfirmed text.
      withContext(Dispatchers.Main) {
        binding.contentMain.transcriptionText.text =
          binding.contentMain.transcriptionText.text.toString()
      }
    }

  private fun updateTranscriptionUI(confirmedText: String, unconfirmedText: String) {
    val spannable = SpannableStringBuilder(confirmedText)
    if (unconfirmedText.isNotEmpty()) {
      if (confirmedText.isNotEmpty()) {
        spannable.append(" ")
      }
      val start = spannable.length
      spannable.append(unconfirmedText)
      spannable.setSpan(
        ForegroundColorSpan(Color.GRAY),
        start,
        spannable.length,
        SpannableStringBuilder.SPAN_EXCLUSIVE_EXCLUSIVE,
      )
    }
    binding.contentMain.transcriptionText.text = spannable
    binding.contentMain.transcriptionScrollView.post {
      binding.contentMain.transcriptionScrollView.fullScroll(ScrollView.FOCUS_DOWN)
    }
  }

  private suspend fun processFileAudio(audioUri: Uri) {
    try {
      contentResolver.openInputStream(audioUri)?.use { inputStream ->
        FileAudioSource(
            inputStream,
            SAMPLING_RATE,
            modelInputInterval,
            FILE_AUDIO_CHUNK_OVERLAP_DURATION,
          )
          .use { audioSource ->
            transcribeAudio(
              audioSource,
              (FILE_AUDIO_CHUNK_OVERLAP_DURATION / modelInputInterval).toFloat(),
              FILE_MAX_LEVENSHTEIN_DISTANCE,
            )
          }
      } ?: throw IllegalArgumentException("Could not open audio file")
      // TODO(b/326662243): Add a visual indicator for completion.
    } catch (e: CancellationException) {
      throw e // Let the coroutine system handle cancellation silently
    } catch (e: Exception) {
      e.printStackTrace()
      withContext(Dispatchers.Main) {
        Toast.makeText(this@MainActivity, "Transcription error: ${e.message}", Toast.LENGTH_LONG)
          .show()
      }
    }
  }

  private fun merge(prev: String, next: String) =
    listOf(prev, next).filter { it.isNotEmpty() }.joinToString(separator = " ")

  private fun copyToClipboard() {
    val text = binding.contentMain.transcriptionText.text.toString()
    if (text.isNotEmpty() && text != getString(R.string.transcription_placeholder)) {
      val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
      val clip = ClipData.newPlainText("Transcription", text)
      clipboard.setPrimaryClip(clip)
      Toast.makeText(this, "Copied to clipboard", Toast.LENGTH_SHORT).show()
    }
  }

  private fun clearTranscription() {
    binding.contentMain.transcriptionText.text = getString(R.string.transcription_placeholder)
  }

  override fun onDestroy() {
    super.onDestroy()

    lifecycleScope.launch(Dispatchers.IO) {
      resetPipeline()
      mediaPlayer?.release()
      mediaPlayer = null
    }
  }

  private companion object {
    const val MODEL_METADATA_PATH = "model_metadata.json"

    const val SAMPLING_RATE = 16000
    val FILE_AUDIO_CHUNK_OVERLAP_DURATION = 2.seconds
    val LIVE_AUDIO_CHUNK_INTERVAL = 5.seconds
    val LIVE_AUDIO_CHUNK_SLIDING_INTERVAL = 1.seconds

    /** Allow up to N errors in Levenshtein distance for word alignment for the unconfirmed text. */
    const val FILE_MAX_LEVENSHTEIN_DISTANCE = 5
    const val LIVE_MAX_LEVENSHTEIN_DISTANCE = 20

    fun hasTpu() = NpuCompatibilityChecker.GoogleTensor.isDeviceSupported()
  }
}
