package com.example.gemma_on_device

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.gemma_on_device.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.util.Locale
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val WAV_HEADER_BYTES = 44L
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var liteRTLMManager: LiteRTLMManager
    private lateinit var modelDownloader: ModelDownloader

    private val messages = mutableListOf<ChatMessage>()
    private var currentModel: ModelConfig? = null

    private var pendingImagePath: String? = null
    private var pendingAudioBytes: ByteArray? = null

    private var audioRecord: AudioRecord? = null
    private var audioFile: File? = null
    private var recordingJob: Job? = null

    @Volatile
    private var isRecording = false
    private var isGenerating = false

    private lateinit var imagePickerLauncher: ActivityResultLauncher<Intent>
    private lateinit var permissionLauncher: ActivityResultLauncher<Array<String>>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        liteRTLMManager = LiteRTLMManager.getInstance(this)
        modelDownloader = ModelDownloader(this)

        setupActivityResultLaunchers()
        setupRecyclerView()
        setupInput()
        setupAttachments()

        binding.iconSettings.setOnClickListener { showSettingsDialog() }

        val defaultModel = ModelDownloader.AVAILABLE_MODELS.first()
        currentModel = defaultModel
        checkAndInitialize(defaultModel)
    }

    private fun setupActivityResultLaunchers() {
        imagePickerLauncher =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
                if (result.resultCode == Activity.RESULT_OK) {
                    result.data?.data?.let { uri -> handleImageUri(uri) }
                }
            }

        permissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
                if (permissions[Manifest.permission.RECORD_AUDIO] == true &&
                    !isRecording &&
                    !isGenerating
                ) {
                    startRecording()
                }
            }
    }

    private fun showSettingsDialog() {
        val options = arrayOf("Select Model", "Set HF Token", "Clear Conversation")
        AlertDialog.Builder(this)
            .setTitle("Settings")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> showModelSelectionDialog()
                    1 -> showTokenDialog()
                    2 -> clearConversation()
                }
            }
            .show()
    }

    private fun showModelSelectionDialog() {
        val modelNames = ModelDownloader.AVAILABLE_MODELS.map { model ->
            val downloaded = if (modelDownloader.isModelDownloaded(model)) " ✓" else ""
            val current = if (model.id == currentModel?.id) " (active)" else ""
            "${model.name}$downloaded$current"
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Select Model")
            .setItems(modelNames) { _, which ->
                val selected = ModelDownloader.AVAILABLE_MODELS[which]
                if (selected.id != currentModel?.id) {
                    currentModel = selected
                    messages.clear()
                    updateMessages()
                    checkAndInitialize(selected)
                }
            }
            .show()
    }

    private fun clearConversation() {
        if (isGenerating || isRecording) return
        messages.clear()
        clearAttachments()
        updateMessages()
        if (liteRTLMManager.isReady()) {
            liteRTLMManager.resetConversation()
            addSystemMessage("Started a new conversation on ${liteRTLMManager.getActiveBackendName()}.")
        } else {
            currentModel?.let { checkAndInitialize(it) }
        }
    }

    private fun showTokenDialog() {
        val input = EditText(this)
        input.hint = "hf_..."
        val currentToken = modelDownloader.getToken()
        if (!currentToken.isNullOrBlank()) {
            input.setText(currentToken)
        }

        AlertDialog.Builder(this)
            .setTitle("Set Hugging Face Token")
            .setMessage("Enter your API token to access gated models.")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val token = input.text.toString().trim()
                modelDownloader.saveToken(token)
                Toast.makeText(this, "Token saved", Toast.LENGTH_SHORT).show()
                if (currentModel != null && !modelDownloader.isModelDownloaded(currentModel!!)) {
                    checkAndInitialize(currentModel!!)
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter()
        binding.recyclerViewMessages.apply {
            adapter = chatAdapter
            layoutManager = LinearLayoutManager(this@MainActivity).apply {
                stackFromEnd = true
            }

            addOnLayoutChangeListener { _, _, _, _, bottom, _, _, _, oldBottom ->
                if (bottom < oldBottom) {
                    binding.recyclerViewMessages.postDelayed({
                        if (messages.isNotEmpty()) {
                            binding.recyclerViewMessages.smoothScrollToPosition(messages.size - 1)
                        }
                    }, 100)
                }
            }
        }
    }

    private fun setupInput() {
        binding.editTextMessage.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                updateSendButtonState()
            }
            override fun afterTextChanged(s: Editable?) = Unit
        })

        binding.buttonSend.setOnClickListener {
            val text = binding.editTextMessage.text.toString().trim()
            if (text.isNotEmpty() || pendingImagePath != null || pendingAudioBytes != null) {
                sendMessage(text.ifEmpty { "Describe what you see/hear." })
                binding.editTextMessage.text?.clear()
            }
        }

        binding.buttonSend.alpha = 0.5f
        binding.buttonSend.isEnabled = false
    }

    private fun setupAttachments() {
        binding.buttonAttachImage.setOnClickListener { pickImage() }
        binding.buttonAttachAudio.setOnClickListener { toggleAudioRecording() }
        binding.textAttachmentStatus.setOnClickListener {
            if (!isRecording && !isGenerating) {
                clearAttachments()
            }
        }
    }

    // ─── Image Handling ─────────────────────────────────────────

    private fun pickImage() {
        if (isGenerating) return
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        imagePickerLauncher.launch(intent)
    }

    private fun handleImageUri(uri: Uri) {
        lifecycleScope.launch {
            try {
                val tempFile = File(cacheDir, "attached_image.jpg")
                withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { input ->
                        tempFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                pendingImagePath = tempFile.absolutePath
                refreshAttachmentBanner()
                updateSendButtonState()
            } catch (e: Exception) {
                Toast.makeText(
                    this@MainActivity,
                    "Failed to attach image: ${e.message}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    // ─── 16 kHz Mono PCM WAV Audio Recording ────────────────────

    private fun toggleAudioRecording() {
        if (isGenerating) return
        if (isRecording) {
            stopRecording()
        } else if (checkAudioPermission()) {
            startRecording()
        }
    }

    private fun checkAudioPermission(): Boolean {
        return if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) {
            true
        } else {
            permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            false
        }
    }

    private fun startRecording() {
        try {
            val sampleRate = 16_000
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val encoding = AudioFormat.ENCODING_PCM_16BIT
            val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, encoding)
            if (minBufferSize <= 0) {
                throw IllegalStateException("Invalid audio buffer size: $minBufferSize")
            }
            val bufferSize = max(minBufferSize, sampleRate / 2)
            val outputFile = File(cacheDir, "recorded_audio.wav")

            @Suppress("DEPRECATION")
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                encoding,
                bufferSize
            )
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                throw IllegalStateException("AudioRecord failed to initialize")
            }

            audioFile = outputFile
            audioRecord = recorder
            isRecording = true
            recorder.startRecording()
            recordingJob = lifecycleScope.launch(Dispatchers.IO) {
                writeWavRecording(outputFile, recorder, sampleRate, bufferSize)
            }

            binding.buttonAttachAudio.alpha = 0.5f
            binding.textAttachmentStatus.text = "Recording 16 kHz audio... tap mic again to stop"
            binding.textAttachmentStatus.visibility = View.VISIBLE
            updateSendButtonState()
        } catch (e: Exception) {
            Log.e(TAG, "Recording failed: ${e.message}", e)
            Toast.makeText(this, "Recording failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopRecording() {
        val recorder = audioRecord
        val file = audioFile
        isRecording = false
        binding.buttonAttachAudio.alpha = 1.0f

        try {
            recorder?.stop()
        } catch (e: Exception) {
            Log.w(TAG, "AudioRecord stop warning: ${e.message}", e)
        }

        lifecycleScope.launch {
            try {
                recordingJob?.join()
                recorder?.release()
                audioRecord = null
                recordingJob = null

                if (file != null && file.exists() && file.length() > WAV_HEADER_BYTES) {
                    pendingAudioBytes = withContext(Dispatchers.IO) { file.readBytes() }
                    refreshAttachmentBanner()
                }
                updateSendButtonState()
            } catch (e: Exception) {
                Log.e(TAG, "Stop recording failed: ${e.message}", e)
            }
        }
    }

    private fun refreshAttachmentBanner() {
        val parts = mutableListOf<String>()
        if (pendingImagePath != null) parts.add("Image")
        if (pendingAudioBytes != null) parts.add("Audio")
        if (parts.isEmpty()) {
            binding.textAttachmentStatus.visibility = View.GONE
        } else {
            binding.textAttachmentStatus.text = "${parts.joinToString(" + ")} attached (tap to clear)"
            binding.textAttachmentStatus.visibility = View.VISIBLE
        }
    }

    private fun writeWavRecording(
        file: File,
        recorder: AudioRecord,
        sampleRate: Int,
        bufferSize: Int
    ) {
        var pcmBytes = 0L
        RandomAccessFile(file, "rw").use { wav ->
            wav.setLength(0)
            writeWavHeader(wav, sampleRate, channels = 1, bitsPerSample = 16, pcmBytes = 0)
            val buffer = ByteArray(bufferSize)
            while (isRecording) {
                val read = recorder.read(buffer, 0, buffer.size)
                if (read > 0) {
                    wav.write(buffer, 0, read)
                    pcmBytes += read
                }
            }
            wav.seek(0)
            writeWavHeader(wav, sampleRate, channels = 1, bitsPerSample = 16, pcmBytes = pcmBytes)
        }
    }

    private fun updateSendButtonState() {
        val hasContent = !binding.editTextMessage.text.isNullOrBlank() ||
            pendingImagePath != null ||
            pendingAudioBytes != null
        val canSend = hasContent && !isGenerating && liteRTLMManager.isReady()
        binding.buttonSend.alpha = if (canSend) 1.0f else 0.5f
        binding.buttonSend.isEnabled = canSend
    }

    // ─── Model Management ───────────────────────────────────────

    private fun checkAndInitialize(modelConfig: ModelConfig) {
        lifecycleScope.launch {
            binding.cardInput.visibility = View.GONE
            binding.layoutLoading.visibility = View.VISIBLE
            binding.textLoadingStatus.text = "Checking ${modelConfig.name}..."
            binding.textModelName.text = modelConfig.name
            binding.textBackendStatus.text = "GPU/CPU"

            if (modelDownloader.isModelDownloaded(modelConfig)) {
                initializeEngine(modelConfig)
            } else {
                showAdbOrDownloadDialog(modelConfig)
            }
        }
    }

    private fun showAdbOrDownloadDialog(modelConfig: ModelConfig) {
        val adbCmd = "adb push ${modelConfig.filename} /sdcard/Android/data/$packageName/files/"
        AlertDialog.Builder(this)
            .setTitle("Model Not Found")
            .setMessage(
                "\"${modelConfig.name}\" is not on this device.\n\n" +
                    "Option 1 (Recommended):\n" +
                    "Push via ADB from your computer:\n  $adbCmd\n\n" +
                    "Option 2:\n" +
                    "Download directly from Hugging Face."
            )
            .setPositiveButton("Download Now") { _, _ ->
                downloadModel(modelConfig)
            }
            .setNeutralButton("I'll ADB Push") { _, _ ->
                binding.textLoadingStatus.text =
                    "Push the model via ADB, then tap here to check again:\n\n$adbCmd"
                binding.layoutLoading.setOnClickListener {
                    binding.layoutLoading.setOnClickListener(null)
                    checkAndInitialize(modelConfig)
                }
            }
            .setNegativeButton("Cancel") { _, _ ->
                binding.layoutLoading.visibility = View.GONE
            }
            .show()
    }

    private fun downloadModel(modelConfig: ModelConfig) {
        lifecycleScope.launch {
            binding.textLoadingStatus.text = "Downloading ${modelConfig.name}..."

            modelDownloader.downloadModel(modelConfig).collect { progress ->
                when (progress) {
                    is DownloadProgress.Started -> {
                        binding.textLoadingStatus.text = "Starting download..."
                    }
                    is DownloadProgress.Progress -> {
                        val mb = progress.downloaded / (1024 * 1024)
                        val totalMb = progress.total / (1024 * 1024)
                        binding.textLoadingStatus.text =
                            "Downloading: ${progress.percentage}% ($mb/$totalMb MB)"
                    }
                    is DownloadProgress.Complete -> {
                        binding.textLoadingStatus.text = "Download complete"
                        initializeEngine(modelConfig)
                    }
                    is DownloadProgress.Error -> {
                        binding.layoutLoading.visibility = View.GONE
                        binding.textLoadingStatus.text = "Error: ${progress.message}"

                        if (progress.message.contains("401") ||
                            progress.message.contains("403") ||
                            progress.message.contains("404")
                        ) {
                            AlertDialog.Builder(this@MainActivity)
                                .setTitle("Download Error")
                                .setMessage("Failed: ${progress.message}\n\nSet a Hugging Face token?")
                                .setPositiveButton("Set Token") { _, _ -> showTokenDialog() }
                                .setNegativeButton("Cancel", null)
                                .show()
                        } else {
                            Toast.makeText(
                                this@MainActivity,
                                "Download failed: ${progress.message}",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }
                }
            }
        }
    }

    private fun initializeEngine(modelConfig: ModelConfig) {
        lifecycleScope.launch {
            binding.layoutLoading.setOnClickListener(null)
            binding.layoutLoading.visibility = View.VISIBLE
            binding.textLoadingStatus.text = "Initializing ${modelConfig.name}...\n(GPU -> CPU)"

            val startTime = System.currentTimeMillis()
            val result = liteRTLMManager.initialize(
                modelDownloader.getModelPath(modelConfig),
                modelConfig.systemPrompt,
                false,
                modelConfig.preferredBackend
            )
            val loadTime = System.currentTimeMillis() - startTime

            binding.layoutLoading.visibility = View.GONE

            if (result.isSuccess) {
                binding.cardInput.visibility = View.VISIBLE
                val backend = liteRTLMManager.getActiveBackendName()
                addSystemMessage("${modelConfig.name} initialized on $backend in ${loadTime}ms.")
                binding.textBackendStatus.text = backend
                binding.textBenchmarkStats.text = "Load: ${loadTime}ms"
            } else {
                val error = result.exceptionOrNull()?.message ?: "Unknown error"
                binding.layoutLoading.visibility = View.VISIBLE
                binding.textLoadingStatus.text = "Initialization failed: $error"
                Toast.makeText(this@MainActivity, "Init failed: $error", Toast.LENGTH_LONG).show()
            }
            updateSendButtonState()
        }
    }

    // ─── Message Sending ────────────────────────────────────────

    private fun sendMessage(text: String) {
        if (isGenerating) return
        isGenerating = true
        updateSendButtonState()

        val imagePath = pendingImagePath
        val audioBytes = pendingAudioBytes
        val attachInfo = buildString {
            if (imagePath != null) append("[Image] ")
            if (audioBytes != null) append("[Audio] ")
            append(text)
        }

        messages.add(ChatMessage(MessageSender.USER, attachInfo))
        var assistantMessageIndex = messages.size
        val assistantMessage = ChatMessage(MessageSender.ASSISTANT, "", isStreaming = true)
        messages.add(assistantMessage)
        updateMessages()
        clearAttachments()

        var firstTokenReceived = false
        var startTime = System.currentTimeMillis()
        var ttft: Long = 0

        lifecycleScope.launch {
            var fullResponse = ""
            val requestStartTime = System.nanoTime()
            var firstTokenTime = 0L

            val onChunk: (String) -> Unit = { messageChunk ->
                if (messageChunk.isNotEmpty()) {
                    if (!firstTokenReceived) {
                        ttft = System.currentTimeMillis() - startTime
                        firstTokenReceived = true
                        startTime = System.currentTimeMillis()
                        firstTokenTime = System.nanoTime()
                    }

                    fullResponse += messageChunk
                    val tokenCount = fullResponse.length / 4 + 1

                    messages[assistantMessageIndex] = assistantMessage.copy(
                        content = fullResponse,
                        isStreaming = true
                    )
                    updateMessages()

                    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                    val speed =
                        if (elapsed > 0) String.format(Locale.US, "%.1f", tokenCount / elapsed) else "0.0"
                    binding.textBenchmarkStats.text = "TTFT: ${ttft}ms | $speed t/s"
                }
            }

            try {
                val outcome = streamTurnWithOverflowRecovery(
                    streamOnce = { emit ->
                        val flow = if (imagePath != null || audioBytes != null) {
                            liteRTLMManager.sendMultimodalMessage(text, imagePath, audioBytes)
                        } else {
                            liteRTLMManager.sendMessage(text)
                        }
                        flow.catch { e -> throw e }.collect { emit(it) }
                    },
                    currentOutput = { fullResponse },
                    resetSession = { liteRTLMManager.resetConversation() },
                    onChunk = onChunk
                )

                if (outcome.retriedInFreshSession) {
                    val notice = ChatMessage(
                        MessageSender.SYSTEM,
                        "Context window filled — automatically started a fresh session and retried your prompt."
                    )
                    messages.add(assistantMessageIndex, notice)
                    assistantMessageIndex += 1
                }

                if (fullResponse.isBlank()) {
                    fullResponse = "No response generated."
                }

                val finalText = if (outcome.truncatedMidStream) {
                    "$fullResponse\n\n[Reached max context window — session reset for your next message.]"
                } else {
                    fullResponse
                }

                messages[assistantMessageIndex] = assistantMessage.copy(
                    content = finalText,
                    isStreaming = false
                )
                updateMessages()

                val endTime = System.nanoTime()
                val ttftMs = if (firstTokenReceived) (firstTokenTime - requestStartTime) / 1_000_000 else 0
                val generationTimeMs = if (firstTokenReceived) (endTime - firstTokenTime) / 1_000_000 else 0
                val tokens = fullResponse.length / 4.0
                val tps = if (generationTimeMs > 0) tokens / (generationTimeMs / 1000.0) else 0.0

                binding.textBenchmarkStats.text = String.format(
                    Locale.US,
                    "TTFT: %dms | %.1f t/s | %d tokens",
                    ttftMs,
                    tps,
                    tokens.toInt()
                )
            } catch (e: Throwable) {
                Log.e(TAG, "Generation failed: ${e.message}", e)
                liteRTLMManager.resetConversation()
                messages[assistantMessageIndex] = assistantMessage.copy(
                    content = "Error: ${e.message}",
                    isStreaming = false
                )
                updateMessages()
            } finally {
                isGenerating = false
                updateSendButtonState()
            }
        }
    }

    private fun clearAttachments() {
        pendingImagePath = null
        pendingAudioBytes = null
        binding.textAttachmentStatus.visibility = View.GONE
        updateSendButtonState()
    }

    private fun addSystemMessage(text: String) {
        messages.add(ChatMessage(MessageSender.SYSTEM, text))
        updateMessages()
    }

    private fun updateMessages() {
        chatAdapter.submitList(messages.toList())
        if (messages.isNotEmpty()) {
            binding.recyclerViewMessages.smoothScrollToPosition(messages.size - 1)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) {
            isRecording = false
            try {
                audioRecord?.stop()
            } catch (_: Exception) {
            }
            audioRecord?.release()
            audioRecord = null
        }
        liteRTLMManager.cleanup()
    }
}

internal fun writeWavHeader(
    wav: RandomAccessFile,
    sampleRate: Int,
    channels: Int,
    bitsPerSample: Int,
    pcmBytes: Long
) {
    val byteRate = sampleRate * channels * bitsPerSample / 8
    val blockAlign = channels * bitsPerSample / 8
    val dataSize = pcmBytes.coerceAtMost(Int.MAX_VALUE.toLong()).toInt()

    wav.writeBytes("RIFF")
    writeLittleEndianInt(wav, 36 + dataSize)
    wav.writeBytes("WAVE")
    wav.writeBytes("fmt ")
    writeLittleEndianInt(wav, 16)
    writeLittleEndianShort(wav, 1)
    writeLittleEndianShort(wav, channels)
    writeLittleEndianInt(wav, sampleRate)
    writeLittleEndianInt(wav, byteRate)
    writeLittleEndianShort(wav, blockAlign)
    writeLittleEndianShort(wav, bitsPerSample)
    wav.writeBytes("data")
    writeLittleEndianInt(wav, dataSize)
}

private fun writeLittleEndianInt(file: RandomAccessFile, value: Int) {
    file.write(value and 0xff)
    file.write((value shr 8) and 0xff)
    file.write((value shr 16) and 0xff)
    file.write((value shr 24) and 0xff)
}

private fun writeLittleEndianShort(file: RandomAccessFile, value: Int) {
    file.write(value and 0xff)
    file.write((value shr 8) and 0xff)
}

internal data class StreamTurnOutcome(
    val retriedInFreshSession: Boolean = false,
    val truncatedMidStream: Boolean = false
)

internal suspend fun streamTurnWithOverflowRecovery(
    streamOnce: suspend (onChunk: (String) -> Unit) -> Unit,
    currentOutput: () -> String,
    resetSession: () -> Boolean,
    onChunk: (String) -> Unit
): StreamTurnOutcome {
    return try {
        streamOnce(onChunk)
        StreamTurnOutcome()
    } catch (e: Throwable) {
        val isRecoverable =
            LiteRTLMManager.isContextOverflowError(e) || LiteRTLMManager.isRoleAlternationError(e)
        if (!isRecoverable) throw e

        resetSession()
        if (currentOutput().isEmpty()) {
            streamOnce(onChunk)
            StreamTurnOutcome(retriedInFreshSession = true)
        } else {
            StreamTurnOutcome(truncatedMidStream = true)
        }
    }
}

