package com.example.gemma_on_device

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.gemma_on_device.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var liteRTLMManager: LiteRTLMManager
    private lateinit var modelDownloader: ModelDownloader
    
    // Conversation history
    private val messages = mutableListOf<ChatMessage>()
    
    private var currentModel: ModelConfig? = null
    
    // Multimodal attachments
    private var pendingImagePath: String? = null
    private var pendingAudioBytes: ByteArray? = null
    
    // Audio recording
    private var mediaRecorder: MediaRecorder? = null
    private var audioFile: File? = null
    private var isRecording = false

    // Activity result launchers
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
        
        // Settings
        binding.iconSettings.setOnClickListener { showSettingsDialog() }

        // Default model: Gemma 4 E2B (first in list)
        val defaultModel = ModelDownloader.AVAILABLE_MODELS.first()
        currentModel = defaultModel
        checkAndInitialize(defaultModel)
    }

    private fun setupActivityResultLaunchers() {
        imagePickerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    handleImageUri(uri)
                }
            }
        }
        
        permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            // Permissions handled — no action needed here as we check at use time
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
        messages.clear()
        updateMessages()
        liteRTLMManager.cleanup()
        currentModel?.let { checkAndInitialize(it) }
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
                Toast.makeText(this, "Token saved!", Toast.LENGTH_SHORT).show()
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
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                val hasContent = !s.isNullOrBlank() || pendingImagePath != null || pendingAudioBytes != null
                binding.buttonSend.alpha = if (hasContent) 1.0f else 0.5f
                binding.buttonSend.isEnabled = hasContent
            }
            override fun afterTextChanged(s: Editable?) {}
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
    }

    // ─── Image Handling ─────────────────────────────────────────
    
    private fun pickImage() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        imagePickerLauncher.launch(intent)
    }

    private fun handleImageUri(uri: Uri) {
        lifecycleScope.launch {
            try {
                // Copy to a temp file the engine can read
                val tempFile = File(cacheDir, "attached_image.jpg")
                withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { input ->
                        tempFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                pendingImagePath = tempFile.absolutePath
                binding.textAttachmentStatus.text = "📷 Image attached"
                binding.textAttachmentStatus.visibility = View.VISIBLE
                updateSendButtonState()
                Toast.makeText(this@MainActivity, "Image attached", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Failed to attach image: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ─── Audio Recording ────────────────────────────────────────

    private fun toggleAudioRecording() {
        if (isRecording) {
            stopRecording()
        } else {
            if (checkAudioPermission()) {
                startRecording()
            }
        }
    }

    private fun checkAudioPermission(): Boolean {
        return if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
            == PackageManager.PERMISSION_GRANTED) {
            true
        } else {
            permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            false
        }
    }

    private fun startRecording() {
        try {
            audioFile = File(cacheDir, "recorded_audio.m4a")
            mediaRecorder = (if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                MediaRecorder(this)
            } else {
                @Suppress("DEPRECATION")
                MediaRecorder()
            }).apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setAudioSamplingRate(16000)
                setOutputFile(audioFile?.absolutePath)
                prepare()
                start()
            }
            isRecording = true
            binding.buttonAttachAudio.alpha = 0.5f
            binding.textAttachmentStatus.text = "🎙️ Recording... tap again to stop"
            binding.textAttachmentStatus.visibility = View.VISIBLE
            Toast.makeText(this, "Recording started", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("MainActivity", "Recording failed: ${e.message}", e)
            Toast.makeText(this, "Recording failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null
            isRecording = false
            binding.buttonAttachAudio.alpha = 1.0f
            
            // Read audio bytes
            audioFile?.let { file ->
                if (file.exists()) {
                    pendingAudioBytes = file.readBytes()
                    binding.textAttachmentStatus.text = "📷🎙️ Audio recorded".let {
                        val parts = mutableListOf<String>()
                        if (pendingImagePath != null) parts.add("📷 Image")
                        parts.add("🎙️ Audio")
                        parts.joinToString(" + ") + " attached"
                    }
                    binding.textAttachmentStatus.visibility = View.VISIBLE
                    updateSendButtonState()
                    Toast.makeText(this, "Audio recorded", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Stop recording failed: ${e.message}", e)
            Toast.makeText(this, "Stop recording failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateSendButtonState() {
        val hasContent = !binding.editTextMessage.text.isNullOrBlank() || 
                         pendingImagePath != null || 
                         pendingAudioBytes != null
        binding.buttonSend.alpha = if (hasContent) 1.0f else 0.5f
        binding.buttonSend.isEnabled = hasContent
    }
    
    // ─── Model Management ───────────────────────────────────────

    private fun checkAndInitialize(modelConfig: ModelConfig) {
        lifecycleScope.launch {
            binding.cardInput.visibility = View.GONE
            binding.layoutLoading.visibility = View.VISIBLE
            binding.textLoadingStatus.text = "Checking ${modelConfig.name}..."
            binding.textModelName.text = modelConfig.name
            
            if (modelDownloader.isModelDownloaded(modelConfig)) {
                initializeEngine(modelConfig)
            } else {
                // Show instructions to push via ADB first
                showAdbOrDownloadDialog(modelConfig)
            }
        }
    }

    private fun showAdbOrDownloadDialog(modelConfig: ModelConfig) {
        AlertDialog.Builder(this)
            .setTitle("Model Not Found")
            .setMessage(
                "\"${modelConfig.name}\" is not on this device.\n\n" +
                "Option 1 (Recommended for large models):\n" +
                "Push via ADB from your PC:\n" +
                "  adb push ${modelConfig.filename} /sdcard/Download/\n\n" +
                "Option 2:\n" +
                "Download directly on device (may be slow for large files)."
            )
            .setPositiveButton("Download Now") { _, _ ->
                downloadModel(modelConfig)
            }
            .setNeutralButton("I'll ADB Push") { _, _ ->
                binding.textLoadingStatus.text = "Push the model via ADB, then reopen the app.\n\nadb push ${modelConfig.filename} /sdcard/Download/"
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
                        binding.textLoadingStatus.text = "Downloading: ${progress.percentage}% ($mb/$totalMb MB)"
                    }
                    is DownloadProgress.Complete -> {
                        binding.textLoadingStatus.text = "Download complete!"
                        initializeEngine(modelConfig)
                    }
                    is DownloadProgress.Error -> {
                        binding.layoutLoading.visibility = View.GONE
                        binding.textLoadingStatus.text = "Error: ${progress.message}"
                        
                        if (progress.message.contains("401") || progress.message.contains("403") || progress.message.contains("404")) {
                             AlertDialog.Builder(this@MainActivity)
                                .setTitle("Download Error")
                                .setMessage("Failed: ${progress.message}.\n\nDo you need to set a Hugging Face Token?")
                                .setPositiveButton("Set Token") { _, _ -> showTokenDialog() }
                                .setNegativeButton("Cancel", null)
                                .show()
                        } else {
                             Toast.makeText(this@MainActivity, "Download failed: ${progress.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            }
        }
    }
    
    private fun initializeEngine(modelConfig: ModelConfig) {
        lifecycleScope.launch {
            binding.layoutLoading.visibility = View.VISIBLE
            binding.textLoadingStatus.text = "Initializing ${modelConfig.name}...\n(NPU → GPU → CPU)"
            
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
        }
    }
    
    // ─── Message Sending ────────────────────────────────────────

    private fun sendMessage(text: String) {
        // Build display text showing what's attached
        val attachInfo = buildString {
            if (pendingImagePath != null) append("[📷 Image] ")
            if (pendingAudioBytes != null) append("[🎙️ Audio] ")
            append(text)
        }
        
        val userMessage = ChatMessage(MessageSender.USER, attachInfo)
        messages.add(userMessage)
        updateMessages()
        
        val assistantMessageIndex = messages.size
        val assistantMessage = ChatMessage(MessageSender.ASSISTANT, "", isStreaming = true)
        messages.add(assistantMessage)
        updateMessages()
        
        // Capture and clear attachments
        val imagePath = pendingImagePath
        val audioBytes = pendingAudioBytes
        clearAttachments()
        
        var firstTokenReceived = false
        var startTime = System.currentTimeMillis()
        var ttft: Long = 0
        var tokenCount = 0

        lifecycleScope.launch {
            var fullResponse = ""
            val requestStartTime = System.nanoTime()
            var firstTokenTime = 0L
            
            try {
                val flow = if (imagePath != null || audioBytes != null) {
                    liteRTLMManager.sendMultimodalMessage(text, imagePath, audioBytes)
                } else {
                    liteRTLMManager.sendMessage(text)
                }
                
                flow.catch { e ->
                        messages[assistantMessageIndex] = assistantMessage.copy(
                            content = "Error: ${e.message}",
                            isStreaming = false
                        )
                        updateMessages()
                    }
                    .collect { messageChunk ->
                        if (!firstTokenReceived) {
                            ttft = System.currentTimeMillis() - startTime
                            firstTokenReceived = true
                            startTime = System.currentTimeMillis()
                            firstTokenTime = System.nanoTime()
                        }
                        
                        val chunkText = messageChunk.toString()
                        fullResponse += chunkText
                        tokenCount = fullResponse.length / 4 + 1
                        
                        messages[assistantMessageIndex] = assistantMessage.copy(
                            content = fullResponse,
                            isStreaming = true
                        )
                        chatAdapter.notifyItemChanged(assistantMessageIndex)
                        binding.recyclerViewMessages.smoothScrollToPosition(assistantMessageIndex)
                        
                        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                        val speed = if (elapsed > 0) String.format("%.1f", tokenCount / elapsed) else "0"
                        
                        binding.textBenchmarkStats.text = "TTFT: ${ttft}ms | ${speed} t/s"
                    }
                
                // Finalize
                messages[assistantMessageIndex] = assistantMessage.copy(
                    content = fullResponse,
                    isStreaming = false
                )
                updateMessages()
                
                val endTime = System.nanoTime()
                val ttftMs = (firstTokenTime - requestStartTime) / 1_000_000
                val generationTimeMs = (endTime - firstTokenTime) / 1_000_000
                val tokens = fullResponse.length / 4.0
                val tps = if (generationTimeMs > 0) (tokens / (generationTimeMs / 1000.0)) else 0.0
                
                binding.textBenchmarkStats.text = String.format(
                    "TTFT: %dms | %.1f t/s | %d tokens", 
                    ttftMs, tps, tokens.toInt()
                )
                
            } catch (e: Exception) {
                messages[assistantMessageIndex] = assistantMessage.copy(
                    content = "Error: ${e.message}",
                    isStreaming = false
                )
                updateMessages()
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
            try {
                mediaRecorder?.apply { stop(); release() }
            } catch (_: Exception) {}
        }
        liteRTLMManager.cleanup()
    }
}
