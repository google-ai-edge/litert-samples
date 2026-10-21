package com.example.gemma3_npu

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.gemma3_npu.databinding.ActivityMainBinding
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import java.util.Locale

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var liteRTLMManager: LiteRTLMManager
    private lateinit var modelDownloader: ModelDownloader

    private val messages = mutableListOf<ChatMessage>()
    private var isGenerating = false
    private var isDownloading = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        liteRTLMManager = LiteRTLMManager.getInstance(this)
        modelDownloader = ModelDownloader(this)

        setupRecyclerView()
        setupInput()
        checkAndInitializeModel()
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
            if (text.isNotEmpty()) {
                sendMessage(text)
                binding.editTextMessage.text?.clear()
            }
        }

        binding.buttonResetChat.setOnClickListener {
            resetChatSession()
        }

        binding.buttonDownloadModel.setOnClickListener {
            downloadModel(ModelDownloader.AVAILABLE_MODELS.first())
        }

        binding.buttonRetryModel.setOnClickListener {
            checkAndInitializeModel()
        }

        binding.buttonSend.alpha = 0.5f
        binding.buttonSend.isEnabled = false
    }

    private fun updateSendButtonState() {
        val hasContent = !binding.editTextMessage.text.isNullOrBlank()
        val canSend = hasContent && !isGenerating && liteRTLMManager.isReady()
        binding.buttonSend.alpha = if (canSend) 1.0f else 0.5f
        binding.buttonSend.isEnabled = canSend
        binding.buttonResetChat.isEnabled = !isGenerating && !isDownloading
    }

    private fun resetChatSession() {
        if (isGenerating || isDownloading) return
        if (!liteRTLMManager.isReady()) {
            checkAndInitializeModel()
            return
        }
        messages.clear()
        liteRTLMManager.resetConversation()
        addSystemMessage("Started a new conversation on ${liteRTLMManager.getActiveBackendName()}.")
        binding.textBenchmarkStats.text = "Ready"
    }

    // ─── Model Provisioning & Initialization ────────────────────

    private fun checkAndInitializeModel() {
        val modelConfig = ModelDownloader.AVAILABLE_MODELS.first()
        binding.textModelName.text = modelConfig.name
        binding.textBackendStatus.text = "NPU"

        if (modelDownloader.isModelDownloaded(modelConfig)) {
            initializeEngine(modelConfig)
            return
        }

        binding.cardInput.visibility = View.GONE
        binding.layoutLoading.visibility = View.VISIBLE
        binding.progressLoading.visibility = View.GONE
        binding.layoutMissingActions.visibility = View.VISIBLE
        binding.textBenchmarkStats.text = "Model missing"
        binding.textLoadingStatus.text =
            "Model package not found on device.\n\nPush via ADB (recommended):\n${modelDownloader.getAdbPushCommand(modelConfig)}\n\nOr download directly from Hugging Face below."
        updateSendButtonState()
    }

    private fun downloadModel(modelConfig: ModelConfig) {
        if (isDownloading) return
        isDownloading = true
        binding.progressLoading.visibility = View.VISIBLE
        binding.layoutMissingActions.visibility = View.GONE
        binding.textBenchmarkStats.text = "Downloading..."
        updateSendButtonState()

        lifecycleScope.launch {
            modelDownloader.downloadModel(modelConfig).collect { progress ->
                when (progress) {
                    is DownloadProgress.Started -> {
                        binding.textLoadingStatus.text = "Connecting to Hugging Face..."
                    }
                    is DownloadProgress.Progress -> {
                        val mb = progress.downloaded / (1024 * 1024)
                        val totalMb = progress.total / (1024 * 1024)
                        binding.textLoadingStatus.text =
                            "Downloading ${modelConfig.filename}\n${progress.percentage}% ($mb / $totalMb MB)"
                    }
                    is DownloadProgress.Complete -> {
                        isDownloading = false
                        initializeEngine(modelConfig)
                    }
                    is DownloadProgress.Error -> {
                        isDownloading = false
                        binding.progressLoading.visibility = View.GONE
                        binding.layoutMissingActions.visibility = View.VISIBLE
                        binding.textBenchmarkStats.text = "Download failed"
                        binding.textLoadingStatus.text =
                            "Download failed: ${progress.message}\n\nPush via ADB instead:\n${modelDownloader.getAdbPushCommand(modelConfig)}"
                        updateSendButtonState()

                        if (progress.message.contains("401") || progress.message.contains("403")) {
                            showTokenDialog(modelConfig)
                        }
                    }
                }
            }
        }
    }

    private fun showTokenDialog(modelConfig: ModelConfig) {
        val input = EditText(this).apply {
            hint = "hf_..."
            modelDownloader.getToken()?.takeIf { it.isNotBlank() }?.let { setText(it) }
        }
        AlertDialog.Builder(this)
            .setTitle("Hugging Face Access Token")
            .setMessage("This repository requires authentication. Enter your Hugging Face read token to retry.")
            .setView(input)
            .setPositiveButton("Save & Retry") { _, _ ->
                modelDownloader.saveToken(input.text.toString().trim())
                downloadModel(modelConfig)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun initializeEngine(modelConfig: ModelConfig) {
        lifecycleScope.launch {
            binding.cardInput.visibility = View.GONE
            binding.layoutLoading.visibility = View.VISIBLE
            binding.progressLoading.visibility = View.VISIBLE
            binding.layoutMissingActions.visibility = View.GONE
            binding.textLoadingStatus.text = "Initializing ${modelConfig.name}..."

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
                binding.progressLoading.visibility = View.GONE
                binding.layoutMissingActions.visibility = View.VISIBLE
                binding.textLoadingStatus.text = "Initialization failed:\n$error"
                Toast.makeText(this@MainActivity, "Init failed: $error", Toast.LENGTH_LONG).show()
            }
            updateSendButtonState()
        }
    }

    // ─── Message Streaming & Context Overflow Recovery ──────────

    private fun sendMessage(text: String) {
        if (isGenerating) {
            Log.w(TAG, "Ignoring send while generation is already running")
            return
        }
        isGenerating = true
        updateSendButtonState()

        messages.add(ChatMessage(MessageSender.USER, text))
        val assistantMessageIndex = messages.size
        messages.add(ChatMessage(MessageSender.ASSISTANT, "", isStreaming = true))
        updateMessages()

        lifecycleScope.launch {
            try {
                streamTurnWithOverflowRecovery(
                    text = text,
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
            liteRTLMManager.sendMessage(text)
                .catch { e -> throw e }
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

            // Reset the native conversation so static KV-cache overflow or stream failures
            // never leave consecutive user roles in the chat template on subsequent turns.
            liteRTLMManager.resetConversation()

            if (isContextIssue && !firstTokenReceived && allowAutoRetry) {
                Log.i(TAG, "Static NPU KV-cache full on prefill; retrying prompt in fresh session")
                streamTurnWithOverflowRecovery(
                    text = text,
                    assistantMessageIndex = assistantMessageIndex,
                    allowAutoRetry = false
                )
                return
            }

            val finalContent = if (isContextIssue && fullResponse.isNotBlank()) {
                "$fullResponse\n\n[Reached 1,280-token NPU context limit — started a fresh session for your next message.]"
            } else if (isContextIssue) {
                "Reached the 1,280-token NPU context window limit. Started a fresh conversation — please send your prompt again."
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
        liteRTLMManager.cleanup()
    }
}
