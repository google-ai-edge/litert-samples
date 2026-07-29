package com.google.googletensortpu.googleTensorTPUApp

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.FileInputStream
import java.io.FileOutputStream
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.view.LayoutInflater
import android.view.inputmethod.InputMethodManager
import android.webkit.MimeTypeMap
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.CheckBox
import android.app.AlertDialog
import android.database.Cursor
import android.provider.OpenableColumns
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var chatAdapter: ChatAdapter
    private lateinit var sendButton: ImageView
    private lateinit var attachButton: ImageView
    private lateinit var selectedImageView: ImageView
    private lateinit var messageEditText: EditText
    private lateinit var spinnerModel: Spinner
    private lateinit var textBackendStatus: TextView
    private lateinit var textBenchmarkStats: TextView
    private lateinit var recyclerView: RecyclerView
    private lateinit var recordingIndicatorLayout: View
    private lateinit var stopRecordingButton: MaterialButton

    private val messagesList = mutableListOf<ChatMessage>()
    private var activeHistoryStartIndex = 0
    private var selectedImageBitmap: Bitmap? = null
    private var pendingImagePath: String? = null
    private var pendingAudioPath: String? = null

    private lateinit var modelResolver: ModelResolver
    private lateinit var liteRTLMManager: LiteRTLMManager
    private var selectedModel: ModelConfig? = null
    private var isGenerating = false
    private var isModelLoaded = false

    private lateinit var cameraContainer: FrameLayout
    private lateinit var viewFinder: androidx.camera.view.PreviewView
    private lateinit var btnCancelCamera: ImageView
    private lateinit var btnCaptureCamera: ImageView
    private lateinit var btnSwitchCamera: ImageView

    private var imageCapture: androidx.camera.core.ImageCapture? = null
    private var lensFacing = androidx.camera.core.CameraSelector.LENS_FACING_BACK
    private lateinit var cameraExecutor: java.util.concurrent.ExecutorService

    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            lifecycleScope.launch(Dispatchers.IO) {
                val bitmap = loadBitmapFromUri(uri)
                if (bitmap != null) {
                    withContext(Dispatchers.Main) {
                        selectedImageBitmap = bitmap
                        selectedImageView.setImageBitmap(bitmap)
                        selectedImageView.visibility = View.VISIBLE
                    }
                    saveBitmapToPendingImagePath(bitmap)
                }
            }
        }
    }

    private val requestCameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            showCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    private val pickAudio = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            copyUriToPendingAudioPath(uri)
            selectedImageView.setImageResource(android.R.drawable.ic_media_play)
            selectedImageView.visibility = View.VISIBLE
        }
    }

    private val pickModelFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            showUploadModelDialog(uri)
        }
    }

    private var isRecording = false
    private var recordingThread: Thread? = null

    private val requestRecordAudioPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            toggleRecording()
        } else {
            Toast.makeText(this, "Microphone permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    private fun toggleRecording() {
        if (isRecording) {
            isRecording = false
            Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
        } else {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                requestRecordAudioPermission.launch(Manifest.permission.RECORD_AUDIO)
                return
            }
            startRecording()
        }
    }

    private fun startRecording() {
        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        val bufferSize = if (minBufferSize == AudioRecord.ERROR_BAD_VALUE || minBufferSize == AudioRecord.ERROR) {
            sampleRate * 2
        } else minBufferSize

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return
        }

        val audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, bufferSize)
        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            Toast.makeText(this, "Failed to initialize AudioRecord", Toast.LENGTH_SHORT).show()
            return
        }

        val tempWav = File(cacheDir, "mic_record.wav")
        val pcmFile = File(cacheDir, "mic_record.pcm")
        pendingAudioPath = tempWav.absolutePath
        isRecording = true
        updateSendButtonState()

        recordingThread = Thread {
            try {
                val os = FileOutputStream(pcmFile)
                val buffer = ByteArray(bufferSize)
                audioRecord.startRecording()
                runOnUiThread {
                    recordingIndicatorLayout.visibility = View.VISIBLE
                }
                while (isRecording) {
                    val read = audioRecord.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        os.write(buffer, 0, read)
                    }
                }
                os.close()
                audioRecord.stop()
                audioRecord.release()
                
                writeWavFile(pcmFile, tempWav, sampleRate, 1, 16)
                pcmFile.delete()
                
                runOnUiThread {
                    recordingIndicatorLayout.visibility = View.GONE
                    selectedImageView.setImageResource(android.R.drawable.ic_media_play)
                    selectedImageView.visibility = View.VISIBLE
                    updateSendButtonState()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Recording failed", e)
                runOnUiThread {
                    recordingIndicatorLayout.visibility = View.GONE
                    isRecording = false
                    updateSendButtonState()
                }
            }
        }
        recordingThread?.start()
    }

    private fun writeWavFile(inPcm: File, outWav: File, sampleRate: Int, channels: Int, bitRate: Int) {
        val inStream = FileInputStream(inPcm)
        val outStream = FileOutputStream(outWav)
        val totalAudioLen = inStream.channel.size()
        val totalDataLen = totalAudioLen + 36
        val byteRate = sampleRate * channels * bitRate / 8

        val header = ByteArray(44)
        header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte(); header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte(); header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
        header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0
        header[20] = 1; header[21] = 0
        header[22] = channels.toByte(); header[23] = 0
        header[24] = (sampleRate and 0xff).toByte(); header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte(); header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte(); header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte(); header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = (channels * bitRate / 8).toByte(); header[33] = 0
        header[34] = bitRate.toByte(); header[35] = 0
        header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte(); header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte(); header[43] = ((totalAudioLen shr 24) and 0xff).toByte()

        outStream.write(header, 0, 44)
        val data = ByteArray(4096)
        var read: Int
        while (inStream.read(data).also { read = it } != -1) {
            outStream.write(data, 0, read)
        }
        inStream.close()
        outStream.close()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)

        try {
            System.load("/vendor/lib64/libedgetpu_litert.so")
            Log.i("MainActivity", "Successfully loaded libedgetpu_litert.so directly!")
            System.load("/vendor/lib64/libedgetpu_util.so")
            Log.i("MainActivity", "Successfully loaded libedgetpu_util.so directly!")
        } catch (e: Throwable) {
            Log.e("MainActivity", "FAILED TO LOAD library: ${e.message}", e)
        }

        ModelResolver.loadCustomModels(this)
        modelResolver = ModelResolver(this)
        liteRTLMManager = LiteRTLMManager.getInstance(this)

        setContentView(R.layout.activity_main)

        recyclerView = findViewById(R.id.chatRecyclerView)
        messageEditText = findViewById(R.id.messageEditText)
        sendButton = findViewById(R.id.sendButton)
        attachButton = findViewById(R.id.attachButton)
        selectedImageView = findViewById(R.id.selectedImageView)
        spinnerModel = findViewById(R.id.spinnerModel)
        textBackendStatus = findViewById(R.id.textBackendStatus)
        textBenchmarkStats = findViewById(R.id.textBenchmarkStats)
        recordingIndicatorLayout = findViewById(R.id.recordingIndicatorLayout)
        stopRecordingButton = findViewById(R.id.stopRecordingButton)

        stopRecordingButton.setOnClickListener {
            toggleRecording()
        }

        cameraExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

        cameraContainer = findViewById(R.id.cameraContainer)
        viewFinder = findViewById(R.id.viewFinder)
        btnCancelCamera = findViewById(R.id.btnCancelCamera)
        btnCaptureCamera = findViewById(R.id.btnCaptureCamera)
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera)

        btnCancelCamera.setOnClickListener {
            hideCamera()
        }

        btnCaptureCamera.setOnClickListener {
            takePhoto()
        }

        btnSwitchCamera.setOnClickListener {
            lensFacing = if (lensFacing == androidx.camera.core.CameraSelector.LENS_FACING_BACK) {
                androidx.camera.core.CameraSelector.LENS_FACING_FRONT
            } else {
                androidx.camera.core.CameraSelector.LENS_FACING_BACK
            }
            startCamera()
        }

        if (savedInstanceState != null) {
            pendingImagePath = savedInstanceState.getString("pendingImagePath")
            pendingAudioPath = savedInstanceState.getString("pendingAudioPath")
            
            val texts = savedInstanceState.getStringArray("chatTexts")
            val isUsers = savedInstanceState.getBooleanArray("chatIsUsers")
            val audioPaths = savedInstanceState.getStringArray("chatAudioPaths")
            val imagePaths = savedInstanceState.getStringArray("chatImagePaths")
            if (texts != null && isUsers != null && texts.size == isUsers.size) {
                for (i in texts.indices) {
                    val audioPath = audioPaths?.get(i)?.takeIf { it.isNotEmpty() }
                    val imagePath = imagePaths?.get(i)?.takeIf { it.isNotEmpty() }
                    val bitmap = if (imagePath != null) {
                        try {
                            BitmapFactory.decodeFile(imagePath)
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to restore message bitmap from: $imagePath", e)
                            null
                        }
                    } else {
                        null
                    }
                    messagesList.add(ChatMessage(texts[i], isUsers[i], bitmap, audioPath, imagePath))
                }
            }
            val selectedModelId = savedInstanceState.getString("selectedModelId")
            if (selectedModelId != null) {
                ModelResolver.AVAILABLE_MODELS.find { it.id == selectedModelId }?.let {
                    selectedModel = it
                }
            }
            
            pendingImagePath?.let { path ->
                lifecycleScope.launch(Dispatchers.IO) {
                    val bitmap = BitmapFactory.decodeFile(path)
                    if (bitmap != null) {
                        withContext(Dispatchers.Main) {
                            selectedImageBitmap = bitmap
                            selectedImageView.setImageBitmap(bitmap)
                            selectedImageView.visibility = View.VISIBLE
                        }
                    }
                }
            }

            pendingAudioPath?.let {
                selectedImageView.setImageResource(android.R.drawable.ic_media_play)
                selectedImageView.visibility = View.VISIBLE
            }
        }

        setupRecyclerView()
        setupModelSelector()
        setupInput()

        // Handle Status Bar padding for edge-to-edge immersive layout
        val headerBar = findViewById<View>(R.id.headerBar)
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(headerBar) { view, windowInsets ->
            val insets = windowInsets.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars())
            val originalPaddingTop = view.tag as? Int ?: view.paddingTop.also { view.tag = it }
            view.setPadding(
                view.paddingLeft,
                originalPaddingTop + insets.top,
                view.paddingRight,
                view.paddingBottom
            )
            windowInsets
        }

        // Handle Navigation Bar margin for edge-to-edge layout
        val inputCard = findViewById<View>(R.id.inputCard)
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(inputCard) { view, windowInsets ->
            val insets = windowInsets.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars() or androidx.core.view.WindowInsetsCompat.Type.ime())
            val params = view.layoutParams as android.view.ViewGroup.MarginLayoutParams
            val originalMarginBottom = view.tag as? Int ?: params.bottomMargin.also { view.tag = it }
            params.bottomMargin = originalMarginBottom + insets.bottom
            view.layoutParams = params
            windowInsets
        }
        
        checkAndInitializeModel()
    }

    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter(messagesList)
        recyclerView.adapter = chatAdapter
        recyclerView.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
    }

    inner class ModelSpinnerAdapter(context: Context, objects: MutableList<String>) :
        ArrayAdapter<String>(context, R.layout.item_spinner_model, android.R.id.text1, objects) {

        override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = convertView ?: LayoutInflater.from(context).inflate(R.layout.item_spinner_dropdown_custom, parent, false)
            val textView = view.findViewById<TextView>(android.R.id.text1)
            val deleteIcon = view.findViewById<ImageView>(R.id.deleteIcon)

            val item = getItem(position)
            textView.text = item

            val isCustomModel = position > 0 && position < count - 1 && !ModelResolver.AVAILABLE_MODELS[position - 1].isAsset

            if (isCustomModel) {
                deleteIcon.visibility = View.VISIBLE
                deleteIcon.setOnClickListener {
                    val modelToDelete = ModelResolver.AVAILABLE_MODELS[position - 1]
                    
                    // Close the spinner dropdown
                    try {
                        val method = Spinner::class.java.getDeclaredMethod("onDetachedFromWindow")
                        method.isAccessible = true
                        method.invoke(spinnerModel)
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }

                    ModelResolver.deleteCustomModel(this@MainActivity, modelToDelete)
                    
                    if (selectedModel?.id == modelToDelete.id) {
                        liteRTLMManager.cleanup()
                        System.gc()
                        selectedModel = null
                        messagesList.clear()
                        chatAdapter.notifyDataSetChanged()
                        clearAttachments()
                        checkAndInitializeModel()
                    }
                    
                    refreshSpinner()
                    Toast.makeText(this@MainActivity, "Model deleted", Toast.LENGTH_SHORT).show()
                }
            } else {
                deleteIcon.visibility = View.GONE
                deleteIcon.setOnClickListener(null)
            }

            return view
        }
    }

    private fun refreshSpinner() {
        val adapter = spinnerModel.adapter as ModelSpinnerAdapter
        adapter.clear()
        adapter.add("Select a model...")
        adapter.addAll(ModelResolver.AVAILABLE_MODELS.map { it.name })
        adapter.add("Upload New Model...")
        adapter.notifyDataSetChanged()
        
        // Ensure the selection stays correct if the list shrinks
        val newIndex = if (selectedModel == null) 0 else ModelResolver.AVAILABLE_MODELS.indexOf(selectedModel) + 1
        if (newIndex >= 0 && newIndex < adapter.count - 1) {
            spinnerModel.setSelection(newIndex)
        } else {
            spinnerModel.setSelection(0)
        }
    }

    private fun setupModelSelector() {
        val adapter = ModelSpinnerAdapter(this, mutableListOf())
        adapter.setDropDownViewResource(R.layout.item_spinner_dropdown_custom)
        spinnerModel.adapter = adapter
        refreshSpinner()
        
        val initialIndex = if (selectedModel == null) 0 else ModelResolver.AVAILABLE_MODELS.indexOf(selectedModel) + 1
        spinnerModel.setSelection(initialIndex)

        var isInitializing = true
        spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (isInitializing) {
                    isInitializing = false
                    return
                }

                if (position == 0) {
                    return
                }
                
                val adapterCount = spinnerModel.adapter.count
                if (position == adapterCount - 1) {
                    // Upload New Model...
                    val prevIndex = if (selectedModel == null) 0 else ModelResolver.AVAILABLE_MODELS.indexOf(selectedModel) + 1
                    spinnerModel.setSelection(prevIndex)
                    pickModelFile.launch(arrayOf("*/*"))
                    return
                }

                val model = ModelResolver.AVAILABLE_MODELS[position - 1]
                if (model.id != selectedModel?.id) {
                    switchModel(model)
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
    }

    private fun setupInput() {
        attachButton.setOnClickListener {
            if (!isGenerating && selectedModel != null) {
                val popup = androidx.appcompat.widget.PopupMenu(this, attachButton)
                val imageInput = popup.menu.add(0, 1, 0, "Image Input")
                val cameraInput = popup.menu.add(0, 2, 0, "Camera")
                val audioInput = popup.menu.add(0, 3, 0, "Audio File Input")
                val micInput = popup.menu.add(0, 4, 0, "Mic Option")

                val supportsImage = selectedModel?.supportsImage == true
                val supportsAudio = selectedModel?.supportsAudio == true
                
                imageInput.isEnabled = supportsImage
                cameraInput.isEnabled = supportsImage
                audioInput.isEnabled = supportsAudio
                micInput.isEnabled = supportsAudio
                
                popup.setOnMenuItemClickListener { item ->
                    when (item.itemId) {
                        1 -> pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                        2 -> {
                            checkCameraPermissionAndStart()
                        }
                        3 -> pickAudio.launch("audio/*")
                        4 -> toggleRecording()
                    }
                    true
                }
                popup.show()
            }
        }

        sendButton.setOnClickListener {
            val query = messageEditText.text.toString().trim()
            if (query.isNotEmpty() || selectedImageBitmap != null || pendingAudioPath != null) {
                val promptText = if (query.isNotEmpty()) {
                    query
                } else if (selectedImageBitmap != null && pendingAudioPath != null) {
                    "Describe this file."
                } else if (selectedImageBitmap != null) {
                    "Describe this image."
                } else if (pendingAudioPath != null) {
                    "Describe this audio."
                } else {
                    "Describe this file."
                }
                sendMessage(promptText)
            }
        }
    }

    private fun updateSendButtonState() {
        val canSend = !isGenerating && isModelLoaded && selectedModel != null && !isRecording
        sendButton.isEnabled = canSend
        sendButton.alpha = if (canSend) 1.0f else 0.5f

        val canAttach = !isGenerating && isModelLoaded && selectedModel != null && !isRecording
        attachButton.isEnabled = canAttach
        attachButton.alpha = if (canAttach) 1.0f else 0.5f

        spinnerModel.isEnabled = !isGenerating && !isRecording
    }

    private fun checkAndInitializeModel() {
        val model = selectedModel
        if (model == null) {
            addSystemMessage("Welcome! Please select or upload a model to begin.")
            isModelLoaded = false
            updateSendButtonState()
            return
        }

        if (modelResolver.isModelAvailable(model)) {
            initializeEngine(model)
        } else {
            addSystemMessage("Model files not found for: ${model.name}.\nPlease push ${model.filename} to ${getExternalFilesDir(null)?.absolutePath}")
            isModelLoaded = false
            updateSendButtonState()
        }
    }

    private fun initializeEngine(model: ModelConfig) {
        activeHistoryStartIndex = 0
        lifecycleScope.launch {
            isModelLoaded = false
            updateSendButtonState()
            textBackendStatus.text = "..."
            textBenchmarkStats.text = "Loading..."
            addSystemMessage("Loading ${model.name}...")
            
            val startTime = System.currentTimeMillis()
            val modelPath = withContext(Dispatchers.IO) {
                modelResolver.resolveModelPath(model)
            }
            
            val history = getConversationHistoryMessages()
            val result = liteRTLMManager.initialize(
                modelPath = modelPath,
                systemPrompt = model.systemPrompt,
                preferredBackend = model.preferredBackend,
                supportsImage = model.supportsImage,
                supportsAudio = model.supportsAudio,
                maxContext = model.maxContext,
                initialMessages = history.takeIf { it.isNotEmpty() }
            )
            
            val loadTime = System.currentTimeMillis() - startTime
            
            if (result.isSuccess) {
                val backend = liteRTLMManager.getActiveBackendName()
                textBackendStatus.text = backend
                textBenchmarkStats.text = "Load: ${loadTime}ms"
                
                // Premium tint coloring that preserves rounded corners of the shape background
                val color = when (backend) {
                    "NPU" -> 0xFF10B981.toInt() // Emerald Green
                    "GPU" -> 0xFF3B82F6.toInt() // Electric Blue
                    else -> 0xFFEF4444.toInt()  // Sunset Red
                }
                textBackendStatus.backgroundTintList = android.content.res.ColorStateList.valueOf(color)
                
                addSystemMessage("${model.name} ready on $backend.")
                isModelLoaded = true
                updateSendButtonState()
            } else {
                val error = result.exceptionOrNull()?.message ?: "Unknown error"
                textBackendStatus.text = "ERR"
                textBenchmarkStats.text = "Init Failed"
                addSystemMessage("Failed to load model: $error")
                Toast.makeText(this@MainActivity, "Initialization failed", Toast.LENGTH_LONG).show()
                isModelLoaded = false
                updateSendButtonState()
            }
        }
    }

    private fun switchModel(modelConfig: ModelConfig) {
        if (isGenerating) return
        activeHistoryStartIndex = 0
        selectedModel = modelConfig
        liteRTLMManager.cleanup()
        System.gc()
        messagesList.clear()
        chatAdapter.notifyDataSetChanged()
        clearAttachments()
        checkAndInitializeModel()
    }

    private fun sendMessage(text: String) {
        if (isGenerating) return
        isGenerating = true
        updateSendButtonState()

        val imagePath = pendingImagePath
        val audioPath = pendingAudioPath

        val uniqueImagePath = copyToUniqueSentFile(imagePath, "sent_image")
        val uniqueAudioPath = copyToUniqueSentFile(audioPath, "sent_audio")

        val userMsg = ChatMessage(text, true, selectedImageBitmap, uniqueAudioPath, uniqueImagePath)
        messagesList.add(userMsg)
        chatAdapter.notifyItemInserted(messagesList.size - 1)
        recyclerView.scrollToPosition(messagesList.size - 1)

        val assistantMsgIndex = messagesList.size
        val assistantMsg = ChatMessage("", false)
        messagesList.add(assistantMsg)
        chatAdapter.notifyItemInserted(messagesList.size - 1)
        recyclerView.scrollToPosition(messagesList.size - 1)

        if (isRecording) {
            Toast.makeText(this@MainActivity, "Stop recording first", Toast.LENGTH_SHORT).show()
            return
        }
        clearAttachments()
        hideKeyboard()

        var startTime = System.currentTimeMillis()
        val requestStartTime = startTime

        lifecycleScope.launch {
            var failed = false
            val flow = if (uniqueImagePath != null || uniqueAudioPath != null) {
                liteRTLMManager.sendMultimodalMessage(text, imagePath = uniqueImagePath, audioPath = uniqueAudioPath)
            } else {
                liteRTLMManager.sendMessage(text)
            }

            collectInferenceFlow(
                flow = flow,
                text = text,
                assistantMsgIndex = assistantMsgIndex,
                startTime = startTime,
                requestStartTime = requestStartTime,
                onFailure = { e ->
                    Log.w(TAG, "Primary inference failed, attempting auto-recovery: ${e.message}")
                    failed = true
                    refreshSessionAndRetry(
                        text = text,
                        imagePath = uniqueImagePath,
                        audioPath = uniqueAudioPath,
                        assistantMsgIndex = assistantMsgIndex
                    )
                }
            )

            if (!failed) {
                isGenerating = false
                updateSendButtonState()
            }
        }
    }

    private fun estimateMessageTokens(msg: ChatMessage): Int {
        var tokens = msg.text.length / 4 + 1
        if (msg.imagePath != null) tokens += 512
        if (msg.audioPath != null) tokens += 256
        return tokens
    }

    private fun getTrimmedConversationHistory(maxBudgetTokens: Int): List<com.google.ai.edge.litertlm.Message> {
        val historyItems = mutableListOf<ChatMessage>()
        val limitIndex = messagesList.size - 2
        for (i in 0 until limitIndex) {
            historyItems.add(messagesList[i])
        }
        
        var totalTokens = (selectedModel?.systemPrompt?.length ?: 0) / 4 + 1
        val keptItems = java.util.ArrayList<ChatMessage>()
        
        var i = historyItems.size - 1
        while (i >= 0) {
            val assistantMsg = historyItems[i]
            if (!assistantMsg.isUser && i > 0) {
                val userMsg = historyItems[i - 1]
                if (userMsg.isUser) {
                    val pairTokens = estimateMessageTokens(userMsg) + estimateMessageTokens(assistantMsg)
                    if (totalTokens + pairTokens <= maxBudgetTokens) {
                        totalTokens += pairTokens
                        keptItems.add(0, assistantMsg)
                        keptItems.add(0, userMsg)
                    } else {
                        break
                    }
                }
                i -= 2
            } else {
                val singleTokens = estimateMessageTokens(assistantMsg)
                if (totalTokens + singleTokens <= maxBudgetTokens) {
                    totalTokens += singleTokens
                    keptItems.add(0, assistantMsg)
                } else {
                    break
                }
                i -= 1
            }
        }
        
        activeHistoryStartIndex = limitIndex - keptItems.size
        Log.i(TAG, "Sliding window optimized: kept ${keptItems.size} messages in model context. Start index set to $activeHistoryStartIndex")
        
        val history = mutableListOf<com.google.ai.edge.litertlm.Message>()
        for (msg in keptItems) {
            if (msg.isUser) {
                val parts = mutableListOf<Content>()
                msg.imagePath?.let { path ->
                    if (File(path).exists()) {
                        parts.add(Content.ImageFile(path))
                    }
                }
                msg.audioPath?.let { path ->
                    if (File(path).exists()) {
                        parts.add(Content.AudioFile(path))
                    }
                }
                parts.add(Content.Text(msg.text))
                history.add(
                    com.google.ai.edge.litertlm.Message.user(
                        Contents.of(*parts.toTypedArray())
                    )
                )
            } else {
                if (msg.text.isNotEmpty() && !msg.text.startsWith("Error:") && !msg.text.startsWith("No response")) {
                    history.add(
                        com.google.ai.edge.litertlm.Message.model(msg.text)
                    )
                }
            }
        }
        return history
    }

    private suspend fun collectInferenceFlow(
        flow: kotlinx.coroutines.flow.Flow<String>,
        text: String,
        assistantMsgIndex: Int,
        startTime: Long,
        requestStartTime: Long,
        onFailure: suspend (Throwable) -> Unit
    ) {
        var firstTokenReceived = false
        var currentStartTime = startTime
        var ttft: Long = 0
        var tokenCount = 0
        var fullResponse = ""
        
        flow.catch { e ->
            onFailure(e)
        }.collect { chunk ->
            if (chunk.isEmpty()) return@collect
            if (!firstTokenReceived) {
                ttft = System.currentTimeMillis() - currentStartTime
                firstTokenReceived = true
                currentStartTime = System.currentTimeMillis()
            }
            
            fullResponse += chunk
            tokenCount = fullResponse.length / 4 + 1
            
            messagesList[assistantMsgIndex] = ChatMessage(fullResponse, false)
            chatAdapter.notifyItemChanged(assistantMsgIndex)
            recyclerView.scrollToPosition(assistantMsgIndex)
            
            val elapsed = (System.currentTimeMillis() - currentStartTime) / 1000.0
            val speed = if (elapsed > 0) String.format("%.1f", tokenCount / elapsed) else "0"
            val estimatedContext = getConversationHistoryTokens() + tokenCount
            val maxContext = selectedModel?.maxContext ?: 2048
            
            val ttftStr = if (firstTokenReceived) "${ttft}ms" else "Calculating..."
            val totalElapsed = (System.currentTimeMillis() - requestStartTime) / 1000.0
            val totalStr = String.format("%.1f s", totalElapsed)
            
            textBenchmarkStats.text = "TTFT: $ttftStr\n$speed t/s | total: $totalStr\ncontext: ~$estimatedContext/$maxContext"
        }
        
        if (fullResponse.isEmpty() && isGenerating) {
            messagesList[assistantMsgIndex] = ChatMessage("No response from model.", false)
            chatAdapter.notifyItemChanged(assistantMsgIndex)
        }
    }

    private fun refreshSessionAndRetry(
        text: String,
        imagePath: String?,
        audioPath: String?,
        assistantMsgIndex: Int
    ) {
        val model = selectedModel ?: return
        
        lifecycleScope.launch {
            messagesList[assistantMsgIndex] = ChatMessage("Optimizing conversation memory...", false)
            chatAdapter.notifyItemChanged(assistantMsgIndex)
            
            // 1. Clean up old engine
            liteRTLMManager.cleanup()
            System.gc()
            
            // 2. Trim history (e.g. budget = 50% of maxContext)
            val budget = (model.maxContext * 0.5).toInt()
            val trimmedHistory = getTrimmedConversationHistory(budget)
            
            // 3. Re-initialize
            val modelPath = withContext(Dispatchers.IO) {
                modelResolver.resolveModelPath(model)
            }
            
            val result = liteRTLMManager.initialize(
                modelPath = modelPath,
                systemPrompt = model.systemPrompt,
                preferredBackend = model.preferredBackend,
                supportsImage = model.supportsImage,
                supportsAudio = model.supportsAudio,
                maxContext = model.maxContext,
                initialMessages = trimmedHistory.takeIf { it.isNotEmpty() }
            )
            
            if (result.isSuccess) {
                // 4. Retry sending the message
                val startTime = System.currentTimeMillis()
                val requestStartTime = startTime
                
                val flow = if (imagePath != null || audioPath != null) {
                    liteRTLMManager.sendMultimodalMessage(text, imagePath = imagePath, audioPath = audioPath)
                } else {
                    liteRTLMManager.sendMessage(text)
                }
                
                collectInferenceFlow(
                    flow = flow,
                    text = text,
                    assistantMsgIndex = assistantMsgIndex,
                    startTime = startTime,
                    requestStartTime = requestStartTime,
                    onFailure = { e ->
                        Log.e(TAG, "Retry inference failed: ${e.message}", e)
                        messagesList[assistantMsgIndex] = ChatMessage("Error after memory optimization: ${e.message}", false)
                        chatAdapter.notifyItemChanged(assistantMsgIndex)
                        isGenerating = false
                        updateSendButtonState()
                    }
                )
                
                isGenerating = false
                updateSendButtonState()
                
            } else {
                val error = result.exceptionOrNull()?.message ?: "Unknown error"
                messagesList[assistantMsgIndex] = ChatMessage("Failed to optimize memory and re-initialize: $error", false)
                chatAdapter.notifyItemChanged(assistantMsgIndex)
                isGenerating = false
                updateSendButtonState()
            }
        }
    }

    private fun copyUriToPendingImagePath(uri: Uri) {
        lifecycleScope.launch {
            try {
                val mimeType = contentResolver.getType(uri)
                val ext = MimeTypeMap.getSingleton().getExtensionFromMimeType(mimeType) ?: "jpg"
                val tempFile = File(cacheDir, "temp_input_image.$ext")
                withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { input ->
                        tempFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                pendingImagePath = tempFile.absolutePath
                Log.i(TAG, "Copied image uri to path: $pendingImagePath")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy image uri", e)
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val maxDim = 1024
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    decoder.isMutableRequired = true
                    var width = info.size.width
                    var height = info.size.height
                    if (width > maxDim || height > maxDim) {
                        val scale = maxDim.toFloat() / Math.max(width, height)
                        width = (width * scale).toInt()
                        height = (height * scale).toInt()
                        decoder.setTargetSize(width, height)
                    }
                }
            } else {
                val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                if (bitmap.width > maxDim || bitmap.height > maxDim) {
                    val scale = maxDim.toFloat() / Math.max(bitmap.width, bitmap.height)
                    Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
                } else {
                    bitmap
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception loading bitmap", e)
            null
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM loading bitmap", e)
            null
        }
    }

    private fun clearAttachments() {
        selectedImageBitmap = null
        pendingImagePath = null
        pendingAudioPath = null
        selectedImageView.visibility = View.GONE
        messageEditText.text.clear()
    }

    private fun copyUriToPendingAudioPath(uri: Uri) {
        lifecycleScope.launch {
            try {
                val mimeType = contentResolver.getType(uri)
                val ext = MimeTypeMap.getSingleton().getExtensionFromMimeType(mimeType) ?: "m4a"
                val tempFile = File(cacheDir, "temp_input_audio.$ext")
                withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { input ->
                        tempFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                pendingAudioPath = tempFile.absolutePath
                Log.i(TAG, "Copied audio uri to path: $pendingAudioPath")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy audio uri", e)
            }
        }
    }

    private fun saveBitmapToPendingImagePath(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val tempFile = File(cacheDir, "camera_image.jpg")
                tempFile.outputStream().use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
                }
                pendingImagePath = tempFile.absolutePath
                Log.i(TAG, "Saved camera bitmap to path: $pendingImagePath")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save camera bitmap", e)
            }
        }
    }

    private fun checkCameraPermissionAndStart() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            showCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun showCamera() {
        cameraContainer.visibility = View.VISIBLE
        startCamera()
    }

    private fun hideCamera() {
        cameraContainer.visibility = View.GONE
        try {
            val cameraProviderFuture = androidx.camera.lifecycle.ProcessCameraProvider.getInstance(this)
            val cameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "Error unbinding camera use cases", e)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = androidx.camera.lifecycle.ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = androidx.camera.core.Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            imageCapture = androidx.camera.core.ImageCapture.Builder()
                .setCaptureMode(androidx.camera.core.ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val cameraSelector = androidx.camera.core.CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageCapture
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
                Toast.makeText(this, "Failed to start camera", Toast.LENGTH_SHORT).show()
            }
        }, androidx.core.content.ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return
        val photoFile = File(cacheDir, "camera_image.jpg")
        val outputOptions = androidx.camera.core.ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions,
            cameraExecutor,
            object : androidx.camera.core.ImageCapture.OnImageSavedCallback {
                override fun onError(exc: androidx.camera.core.ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Photo capture failed", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onImageSaved(output: androidx.camera.core.ImageCapture.OutputFileResults) {
                    val savedUri = Uri.fromFile(photoFile)
                    Log.i(TAG, "Photo capture succeeded: $savedUri")
                    runOnUiThread {
                        cameraContainer.visibility = View.GONE
                        try {
                            val cameraProviderFuture = androidx.camera.lifecycle.ProcessCameraProvider.getInstance(this@MainActivity)
                            val cameraProvider = cameraProviderFuture.get()
                            cameraProvider.unbindAll()
                        } catch (e: Exception) {
                            Log.e(TAG, "Error unbinding camera use cases in onImageSaved", e)
                        }
                        lifecycleScope.launch(Dispatchers.IO) {
                            val bitmap = loadBitmapFromUri(savedUri)
                            if (bitmap != null) {
                                withContext(Dispatchers.Main) {
                                    selectedImageBitmap = bitmap
                                    selectedImageView.setImageBitmap(bitmap)
                                    selectedImageView.visibility = View.VISIBLE
                                }
                                saveBitmapToPendingImagePath(bitmap)
                            }
                        }
                    }
                }
            }
        )
    }

    private fun addSystemMessage(text: String) {
        messagesList.add(ChatMessage("$text", false))
        chatAdapter.notifyItemInserted(messagesList.size - 1)
        recyclerView.scrollToPosition(messagesList.size - 1)
    }

    private fun copyToUniqueSentFile(originalPath: String?, prefix: String): String? {
        if (originalPath == null) return null
        return try {
            val srcFile = File(originalPath)
            if (srcFile.exists()) {
                val destFile = File(cacheDir, "${prefix}_${System.currentTimeMillis()}_${srcFile.name}")
                srcFile.copyTo(destFile, overwrite = true)
                destFile.absolutePath
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy sent file: $originalPath", e)
            null
        }
    }

    private fun getConversationHistoryMessages(): List<com.google.ai.edge.litertlm.Message> {
        val history = mutableListOf<com.google.ai.edge.litertlm.Message>()
        for (i in messagesList.indices) {
            val msg = messagesList[i]
            if (msg.isUser) {
                val parts = mutableListOf<Content>()
                msg.imagePath?.let { path ->
                    if (File(path).exists()) {
                        parts.add(Content.ImageFile(path))
                    }
                }
                msg.audioPath?.let { path ->
                    if (File(path).exists()) {
                        parts.add(Content.AudioFile(path))
                    }
                }
                parts.add(Content.Text(msg.text))
                history.add(
                    com.google.ai.edge.litertlm.Message.user(
                        Contents.of(*parts.toTypedArray())
                    )
                )
            } else {
                if (i > 0 && messagesList[i - 1].isUser && msg.text.isNotEmpty() && !msg.text.startsWith("Error:") && !msg.text.startsWith("No response")) {
                    history.add(
                        com.google.ai.edge.litertlm.Message.model(msg.text)
                    )
                }
            }
        }
        return history
    }

    private fun hideKeyboard() {
        val view = this.currentFocus
        if (view != null) {
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.hideSoftInputFromWindow(view.windowToken, 0)
        }
    }

    private fun showUploadModelDialog(uri: Uri) {
        val filename = getFileName(uri) ?: "custom_model.litertlm"
        
        val dialogView = layoutInflater.inflate(R.layout.dialog_upload_model, null)
        val tvFileName = dialogView.findViewById<TextView>(R.id.dialogModelFileName)
        val etSystemPrompt = dialogView.findViewById<EditText>(R.id.dialogSystemPrompt)
        val etDefaultPrompt = dialogView.findViewById<EditText>(R.id.dialogDefaultPrompt)
        val spinnerBackend = dialogView.findViewById<Spinner>(R.id.dialogPreferredBackend)
        val etMaxContext = dialogView.findViewById<EditText>(R.id.dialogMaxContext)
        val cbImage = dialogView.findViewById<CheckBox>(R.id.dialogSupportsImage)
        val cbAudio = dialogView.findViewById<CheckBox>(R.id.dialogSupportsAudio)

        tvFileName.text = "File: $filename"
        etSystemPrompt.setText("You are a helpful assistant.")
        etDefaultPrompt.setText("help answer based on the provided context")
        etMaxContext.setText("2048")
        
        val backends = arrayOf("NPU", "GPU", "CPU")
        spinnerBackend.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, backends)

        AlertDialog.Builder(this)
            .setTitle("Upload Model")
            .setView(dialogView)
            .setPositiveButton("Add Model") { _, _ ->
                val systemPrompt = etSystemPrompt.text.toString()
                val defaultPrompt = etDefaultPrompt.text.toString()
                val backend = spinnerBackend.selectedItem.toString()
                val supportsImage = cbImage.isChecked
                val supportsAudio = cbAudio.isChecked
                val maxContextStr = etMaxContext.text.toString()
                val maxContext = maxContextStr.toIntOrNull() ?: 2048

                val modelConfig = ModelConfig(
                    id = "custom-${System.currentTimeMillis()}",
                    name = filename,
                    filename = filename,
                    isAsset = false,
                    systemPrompt = systemPrompt,
                    preferredBackend = backend,
                    supportsImage = supportsImage,
                    supportsAudio = supportsAudio,
                    defaultPrompt = defaultPrompt,
                    maxContext = maxContext
                )
                
                copyModelAndAdd(uri, modelConfig)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun getFileName(uri: Uri): String? {
        var result: String? = null
        if (uri.scheme == "content") {
            val cursor: Cursor? = contentResolver.query(uri, null, null, null, null)
            try {
                if (cursor != null && cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index != -1) {
                        result = cursor.getString(index)
                    }
                }
            } finally {
                cursor?.close()
            }
        }
        if (result == null) {
            result = uri.path
            val cut = result?.lastIndexOf('/') ?: -1
            if (cut != -1) {
                result = result?.substring(cut + 1)
            }
        }
        return result
    }

    private fun copyModelAndAdd(uri: Uri, modelConfig: ModelConfig) {
        val progressDialog = AlertDialog.Builder(this)
            .setTitle("Copying Model")
            .setMessage("Please wait while the model is being copied...")
            .setCancelable(false)
            .show()

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val externalDir = getExternalFilesDir(null) ?: File("/sdcard/Android/data/${packageName}/files")
                if (!externalDir.exists()) {
                    externalDir.mkdirs()
                }
                val destFile = File(externalDir, modelConfig.filename)
                
                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(destFile).use { output ->
                        input.copyTo(output)
                    }
                }
                
                withContext(Dispatchers.Main) {
                    progressDialog.dismiss()
                    ModelResolver.saveCustomModel(this@MainActivity, modelConfig)
                    // Refresh spinner
                    refreshSpinner()
                    
                    // Select the newly added model
                    val newIndex = ModelResolver.AVAILABLE_MODELS.indexOf(modelConfig) + 1
                    spinnerModel.setSelection(newIndex)
                    Toast.makeText(this@MainActivity, "Model added successfully", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy model file", e)
                withContext(Dispatchers.Main) {
                    progressDialog.dismiss()
                    Toast.makeText(this@MainActivity, "Failed to copy model", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        pendingImagePath?.let { outState.putString("pendingImagePath", it) }
        pendingAudioPath?.let { outState.putString("pendingAudioPath", it) }
        
        val texts = messagesList.map { it.text }.toTypedArray()
        val isUsers = messagesList.map { it.isUser }.toBooleanArray()
        val audioPaths = messagesList.map { it.audioPath ?: "" }.toTypedArray()
        val imagePaths = messagesList.map { it.imagePath ?: "" }.toTypedArray()
        outState.putStringArray("chatTexts", texts)
        outState.putBooleanArray("chatIsUsers", isUsers)
        outState.putStringArray("chatAudioPaths", audioPaths)
        outState.putStringArray("chatImagePaths", imagePaths)
        selectedModel?.let { outState.putString("selectedModelId", it.id) }
    }

    private fun getConversationHistoryTokens(): Int {
        var totalTokens = (selectedModel?.systemPrompt?.length ?: 0) / 4 + 1
        val limit = messagesList.size - 1
        for (i in activeHistoryStartIndex until limit) {
            if (i in messagesList.indices) {
                totalTokens += estimateMessageTokens(messagesList[i])
            }
        }
        return totalTokens
    }

    override fun onDestroy() {
        super.onDestroy()
        liteRTLMManager.cleanup()
        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }
    }
}