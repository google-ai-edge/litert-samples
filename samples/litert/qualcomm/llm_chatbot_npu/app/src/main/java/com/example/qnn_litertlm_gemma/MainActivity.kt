package com.example.qnn_litertlm_gemma

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
import android.webkit.MimeTypeMap
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.qnn_litertlm_gemma.databinding.ActivityMainBinding
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

    private var pendingImagePath: String? = null
    private var pendingAudioPath: String? = null

    private var audioRecord: AudioRecord? = null
    private var audioFile: File? = null
    private var recordingJob: Job? = null

    @Volatile
    private var isRecording = false
    private var isGenerating = false
    private var selectedModel = ModelDownloader.GEMMA4_NPU

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
        setupModelSelector()
        setupInput()
        setupAttachments()
        checkAndInitializeModel()
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

    private fun setupModelSelector() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            ModelDownloader.AVAILABLE_MODELS.map { it.name }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerModel.adapter = adapter
        binding.spinnerModel.setSelection(ModelDownloader.AVAILABLE_MODELS.indexOf(selectedModel))
        binding.spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val model = ModelDownloader.AVAILABLE_MODELS[position]
                if (model.id != selectedModel.id) {
                    switchModel(model)
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
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
            if (text.isNotEmpty() || pendingImagePath != null || pendingAudioPath != null) {
                sendMessage(text.ifEmpty { selectedModel.defaultPrompt })
                binding.editTextMessage.text?.clear()
            }
        }

        binding.buttonResetChat.setOnClickListener {
            resetChatSession()
        }

        binding.buttonRetryModel.setOnClickListener {
            checkAndInitializeModel()
        }

        binding.textAttachmentStatus.setOnClickListener {
            if (!isRecording) {
                clearAttachments()
            }
        }

        binding.buttonSend.alpha = 0.5f
        binding.buttonSend.isEnabled = false
    }

    private fun setupAttachments() {
        binding.buttonAttachImage.setOnClickListener { pickImage() }
        binding.buttonAttachAudio.setOnClickListener { toggleAudioRecording() }
    }

    private fun resetChatSession() {
        if (isGenerating || isRecording) return
        if (!liteRTLMManager.isReady()) {
            checkAndInitializeModel()
            return
        }
        clearAttachments()
        messages.clear()
        liteRTLMManager.resetConversation()
        addSystemMessage("Started a new conversation on ${liteRTLMManager.getActiveBackendName()}.")
        binding.textBenchmarkStats.text = "Ready"
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
                val mimeType = contentResolver.getType(uri)
                val extension = MimeTypeMap.getSingleton()
                    .getExtensionFromMimeType(mimeType)
                    ?.takeIf { it.isNotBlank() }
                    ?: "img"
                val tempFile = File(cacheDir, "attached_image.$extension")
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

    // ─── Audio Recording ────────────────────────────────────────

    private fun toggleAudioRecording() {
        if (isGenerating) return
        if (!selectedModel.supportsAudio) {
            Toast.makeText(
                this,
                "${selectedModel.name} does not support audio input.",
                Toast.LENGTH_SHORT
            ).show()
            return
        }
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
        updateSendButtonState()

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
                    pendingAudioPath = file.absolutePath
                    refreshAttachmentBanner()
                    updateSendButtonState()
                } else {
                    refreshAttachmentBanner()
                    Toast.makeText(this@MainActivity, "No audio captured", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Stop recording failed: ${e.message}", e)
                Toast.makeText(
                    this@MainActivity,
                    "Stop recording failed: ${e.message}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun refreshAttachmentBanner() {
        val parts = mutableListOf<String>()
        if (pendingImagePath != null) parts.add("Image")
        if (pendingAudioPath != null) parts.add("Audio")
        if (parts.isEmpty()) {
            binding.textAttachmentStatus.visibility = View.GONE
        } else {
            binding.textAttachmentStatus.text =
                "${parts.joinToString(" + ")} attached (tap to clear)"
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

    private fun writeWavHeader(
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

    private fun updateSendButtonState() {
        val hasContent = !binding.editTextMessage.text.isNullOrBlank() ||
            pendingImagePath != null ||
            pendingAudioPath != null
        val canSend = hasContent && !isGenerating && liteRTLMManager.isReady()
        binding.buttonSend.alpha = if (canSend) 1.0f else 0.5f
        binding.buttonSend.isEnabled = canSend
        binding.spinnerModel.isEnabled = !isGenerating && !isRecording
        binding.buttonResetChat.isEnabled = !isGenerating && !isRecording

        val imageEnabled = !isGenerating && selectedModel.supportsImage
        binding.buttonAttachImage.isEnabled = imageEnabled
        binding.buttonAttachImage.alpha = if (imageEnabled) 1.0f else 0.35f

        val audioEnabled = !isGenerating && selectedModel.supportsAudio
        binding.buttonAttachAudio.isEnabled = audioEnabled
        binding.buttonAttachAudio.alpha = when {
            isRecording -> 0.5f
            audioEnabled -> 1.0f
            else -> 0.35f
        }
    }

    // ─── Model Management ───────────────────────────────────────

    private fun checkAndInitializeModel() {
        binding.textBackendStatus.text = "NPU"

        if (modelDownloader.isModelAvailable(selectedModel)) {
            initializeEngine()
            return
        }

        binding.cardInput.visibility = View.GONE
        binding.layoutLoading.visibility = View.VISIBLE
        binding.progressLoading.visibility = View.GONE
        binding.buttonRetryModel.visibility = View.VISIBLE
        binding.textBenchmarkStats.text = "Model missing"
        binding.textLoadingStatus.text =
            "Model not found on device.\n\nPush via ADB and tap Check Again:\n${modelDownloader.getAdbPushCommand(selectedModel)}"
        updateSendButtonState()
    }

    private fun initializeEngine() {
        lifecycleScope.launch {
            val modelConfig = selectedModel
            binding.cardInput.visibility = View.GONE
            binding.layoutLoading.visibility = View.VISIBLE
            binding.progressLoading.visibility = View.VISIBLE
            binding.buttonRetryModel.visibility = View.GONE
            binding.textLoadingStatus.text = "Initializing ${modelConfig.name}..."

            val startTime = System.currentTimeMillis()
            val result = liteRTLMManager.initialize(
                modelDownloader.getModelPath(modelConfig),
                modelConfig.systemPrompt,
                false,
                modelConfig.preferredBackend,
                modelConfig.supportsImage,
                modelConfig.supportsAudio
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
                binding.progressLoading.visibility = View.GONE
                binding.buttonRetryModel.visibility = View.VISIBLE
                binding.textLoadingStatus.text = "Initialization failed:\n$error"
                Toast.makeText(this@MainActivity, "Init failed: $error", Toast.LENGTH_LONG).show()
            }
            updateSendButtonState()
        }
    }

    private fun switchModel(modelConfig: ModelConfig) {
        if (isGenerating || isRecording) {
            binding.spinnerModel.setSelection(ModelDownloader.AVAILABLE_MODELS.indexOf(selectedModel))
            return
        }

        selectedModel = modelConfig
        liteRTLMManager.cleanup()
        messages.clear()
        updateMessages()
        clearAttachments()
        checkAndInitializeModel()
    }

    // ─── Message Sending ────────────────────────────────────────

    private fun sendMessage(text: String) {
        if (isGenerating) {
            Log.w(TAG, "Ignoring send while generation is already running")
            return
        }
        isGenerating = true
        updateSendButtonState()

        val imagePath = pendingImagePath
        val audioPath = if (selectedModel.supportsAudio) pendingAudioPath else null
        val attachInfo = buildString {
            if (imagePath != null) append("[Image] ")
            if (audioPath != null) append("[Audio] ")
            append(text)
        }

        messages.add(ChatMessage(MessageSender.USER, attachInfo))
        val assistantMessageIndex = messages.size
        val assistantPlaceholder = ChatMessage(MessageSender.ASSISTANT, "", isStreaming = true)
        messages.add(assistantPlaceholder)
        updateMessages()
        clearAttachments()

        lifecycleScope.launch {
            try {
                streamTurnWithOverflowRecovery(
                    text = text,
                    imagePath = imagePath,
                    audioPath = audioPath,
                    assistantMessageIndex = assistantMessageIndex,
                    allowAutoRetry = true
                )
            } finally {
                isGenerating = false
                updateSendButtonState()
            }
        }
    }

    private suspend fun streamTurnWithOverflowRecovery(
        text: String,
        imagePath: String?,
        audioPath: String?,
        assistantMessageIndex: Int,
        allowAutoRetry: Boolean
    ) {
        var fullResponse = ""
        var firstTokenReceived = false
        var startTime = System.currentTimeMillis()
        val requestStartTime = System.nanoTime()
        var firstTokenTime = 0L
        var ttft: Long = 0

        try {
            val flow = if (imagePath != null || audioPath != null) {
                liteRTLMManager.sendMultimodalMessage(text, imagePath, audioPath)
            } else {
                liteRTLMManager.sendMessage(text)
            }

            flow.catch { e -> throw e }
                .collect { chunkText ->
                    if (chunkText.isEmpty()) return@collect

                    if (!firstTokenReceived) {
                        ttft = System.currentTimeMillis() - startTime
                        firstTokenReceived = true
                        startTime = System.currentTimeMillis()
                        firstTokenTime = System.nanoTime()
                    }

                    fullResponse += chunkText
                    val tokenCount = fullResponse.length / 4 + 1

                    messages[assistantMessageIndex] = ChatMessage(
                        sender = MessageSender.ASSISTANT,
                        content = fullResponse,
                        isStreaming = true
                    )
                    updateMessages()

                    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                    val speed =
                        if (elapsed > 0) String.format(Locale.US, "%.1f", tokenCount / elapsed) else "0.0"
                    binding.textBenchmarkStats.text = "TTFT: ${ttft}ms | $speed t/s"
                }

            if (fullResponse.isBlank()) {
                fullResponse = "No response generated."
            }

            messages[assistantMessageIndex] = ChatMessage(
                sender = MessageSender.ASSISTANT,
                content = fullResponse,
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
            val isContextIssue =
                LiteRTLMManager.isContextOverflowError(e) || LiteRTLMManager.isRoleAlternationError(e)

            // Always reset the underlying native conversation after a failed turn so the
            // chat template never retains a dangling user message on subsequent turns.
            liteRTLMManager.resetConversation()

            if (isContextIssue && !firstTokenReceived && allowAutoRetry) {
                Log.i(TAG, "Context overflow on prefill; retrying prompt in a fresh conversation")
                streamTurnWithOverflowRecovery(
                    text = text,
                    imagePath = imagePath,
                    audioPath = audioPath,
                    assistantMessageIndex = assistantMessageIndex,
                    allowAutoRetry = false
                )
                return
            }

            val finalContent = if (isContextIssue && fullResponse.isNotBlank()) {
                "$fullResponse\n\n[Reached NPU context limit — started a fresh session for your next message.]"
            } else if (isContextIssue) {
                "Reached the NPU context window limit. Started a fresh conversation — please send your prompt again."
            } else {
                "Error: ${e.message}"
            }

            messages[assistantMessageIndex] = ChatMessage(
                sender = MessageSender.ASSISTANT,
                content = finalContent,
                isStreaming = false
            )
            updateMessages()
        }
    }

    private fun clearAttachments() {
        pendingImagePath = null
        pendingAudioPath = null
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
