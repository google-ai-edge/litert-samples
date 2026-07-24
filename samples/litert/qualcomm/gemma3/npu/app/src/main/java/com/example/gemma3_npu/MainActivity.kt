package com.example.gemma3_npu

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
import android.util.Log
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.webkit.MimeTypeMap
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.gemma3_npu.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
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
    
    // Conversation history
    private val messages = mutableListOf<ChatMessage>()
    
    // State
    private var isGenerating = false



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
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                updateSendButtonState()
            }
            override fun afterTextChanged(s: Editable?) {}
        })
        
        binding.buttonSend.setOnClickListener {
            val text = binding.editTextMessage.text.toString().trim()
            if (text.isNotEmpty()) {
                sendMessage(text)
                binding.editTextMessage.text?.clear()
            }
        }
        
        binding.buttonSend.alpha = 0.5f
        binding.buttonSend.isEnabled = false
    }

    private fun updateSendButtonState() {
        val hasContent = !binding.editTextMessage.text.isNullOrBlank()
        val canSend = hasContent && !isGenerating
        binding.buttonSend.alpha = if (canSend) 1.0f else 0.5f
        binding.buttonSend.isEnabled = canSend
    }
    
    // ─── Model Management ───────────────────────────────────────

    private fun checkAndInitializeModel() {
        val modelConfig = ModelDownloader.AVAILABLE_MODELS.first()
        binding.textModelName.text = modelConfig.name
        binding.textBackendStatus.text = "NPU"

        if (modelDownloader.isModelDownloaded(modelConfig)) {
            initializeEngine()
            return
        }

        binding.cardInput.visibility = View.GONE
        binding.layoutLoading.visibility = View.VISIBLE
        binding.progressLoading.visibility = View.GONE
        binding.textBenchmarkStats.text = "Model missing"
        binding.textLoadingStatus.text = "Please push the model"
    }

    private fun initializeEngine() {
        lifecycleScope.launch {
            val modelConfig = ModelDownloader.AVAILABLE_MODELS.first()
            binding.cardInput.visibility = View.GONE
            binding.layoutLoading.visibility = View.VISIBLE
            binding.progressLoading.visibility = View.VISIBLE
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
                binding.textLoadingStatus.text = "Initialization failed: $error"
                Toast.makeText(this@MainActivity, "Init failed: $error", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    // ─── Message Sending ────────────────────────────────────────

    private fun sendMessage(text: String) {
        if (isGenerating) {
            Log.w(TAG, "Ignoring send while generation is already running")
            return
        }
        isGenerating = true
        updateSendButtonState()

        val userMessage = ChatMessage(MessageSender.USER, text)
        messages.add(userMessage)
        updateMessages()
        
        val assistantMessageIndex = messages.size
        val assistantMessage = ChatMessage(MessageSender.ASSISTANT, "", isStreaming = true)
        messages.add(assistantMessage)
        updateMessages()
        
        var firstTokenReceived = false
        var startTime = System.currentTimeMillis()
        var ttft: Long = 0
        var tokenCount = 0

        lifecycleScope.launch {
            var fullResponse = ""
            val requestStartTime = System.nanoTime()
            var firstTokenTime = 0L
            
            try {
                val flow = liteRTLMManager.sendMessage(text)
                
                Log.i(TAG, "Sending prompt to ${liteRTLMManager.getActiveBackendName()}: ${text.take(80)}")
                flow.catch { e ->
                        Log.e(TAG, "Generation stream failed: ${e.message}", e)
                        throw e
                    }
                    .collect { messageChunk ->
                        if (messageChunk.isEmpty()) {
                            return@collect
                        }
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
                        updateMessages()
                        binding.recyclerViewMessages.smoothScrollToPosition(assistantMessageIndex)
                        
                        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                        val speed = if (elapsed > 0) String.format("%.1f", tokenCount / elapsed) else "0"
                        
                        binding.textBenchmarkStats.text = "TTFT: ${ttft}ms | ${speed} t/s"
                    }
                
                if (fullResponse.isBlank()) {
                    fullResponse = "No response generated."
                    Log.w(TAG, "Generation completed without text")
                }

                // Finalize
                messages[assistantMessageIndex] = assistantMessage.copy(
                    content = fullResponse,
                    isStreaming = false
                )
                Log.i(TAG, "Generation completed with ${fullResponse.length} chars")
                updateMessages()
                
                val endTime = System.nanoTime()
                val ttftMs = if (firstTokenReceived) (firstTokenTime - requestStartTime) / 1_000_000 else 0
                val generationTimeMs = if (firstTokenReceived) (endTime - firstTokenTime) / 1_000_000 else 0
                val tokens = fullResponse.length / 4.0
                val tps = if (generationTimeMs > 0) (tokens / (generationTimeMs / 1000.0)) else 0.0
                
                binding.textBenchmarkStats.text = String.format(
                    "TTFT: %dms | %.1f t/s | %d tokens", 
                    ttftMs, tps, tokens.toInt()
                )
                
            } catch (e: Throwable) {
                Log.e(TAG, "Generation failed: ${e.message}", e)
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
