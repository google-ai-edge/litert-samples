package com.example.fastvlm

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import coil.load
import coil.transform.RoundedCornersTransformation
import com.example.fastvlm.adapter.ChatAdapter
import com.example.fastvlm.databinding.ActivityMainBinding
import com.example.fastvlm.model.EngineStatus
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: com.example.fastvlm.viewmodel.ChatViewModel by viewModels()
    private lateinit var chatAdapter: ChatAdapter

    private var cameraImageUri: Uri? = null

    // ─── Activity Result Launchers ─────────────────────────────

    private val pickMediaLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let { handleImageSelected(it) }
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            cameraImageUri?.let { handleImageSelected(it) }
        }
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchCamera()
        else Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
    }

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchImagePicker()
        else Toast.makeText(this, "Storage permission required", Toast.LENGTH_SHORT).show()
    }

    // ─── Lifecycle ─────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupRecyclerView()
        setupInputBar()
        setupBackendSelector()
        setupButtons()
        observeUiState()

        // Auto-initialize engine on launch
        viewModel.initializeEngine("NPU")
    }

    // ─── RecyclerView Setup ────────────────────────────────────

    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter()
        binding.chatRecyclerView.apply {
            adapter = chatAdapter
            layoutManager = LinearLayoutManager(this@MainActivity).apply {
                stackFromEnd = true
            }
            itemAnimator = null // Disable animations for smoother streaming
        }
    }

    // ─── Input Bar ─────────────────────────────────────────────

    private fun setupInputBar() {
        // Send on button click
        binding.btnSend.setOnClickListener {
            sendMessage()
        }

        // Send on keyboard action
        binding.etMessage.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                sendMessage()
                true
            } else false
        }

        // Attach image
        binding.btnAttachImage.setOnClickListener {
            requestImagePicker()
        }

        // Camera
        binding.btnCamera.setOnClickListener {
            requestCamera()
        }
    }

    // ─── Backend Selector ──────────────────────────────────────

    private fun setupBackendSelector() {
        val chips = listOf(
            binding.chipNpu to "NPU",
            binding.chipGpu to "GPU",
            binding.chipCpu to "CPU"
        )

        chips.forEach { (chip, backend) ->
            chip.setOnClickListener {
                viewModel.switchBackend(backend)
            }
        }
    }

    private fun updateBackendChips(selectedBackend: String) {
        val chips = mapOf(
            "NPU" to binding.chipNpu,
            "GPU" to binding.chipGpu,
            "CPU" to binding.chipCpu
        )

        chips.forEach { (backend, chip) ->
            if (backend == selectedBackend) {
                chip.setBackgroundResource(R.drawable.bg_chip_active)
                chip.setTextColor(ContextCompat.getColor(this, R.color.text_on_accent))
            } else {
                chip.background = null
                chip.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
            }
        }
    }

    // ─── Buttons ───────────────────────────────────────────────

    private fun setupButtons() {
        binding.btnReset.setOnClickListener {
            viewModel.resetConversation()
        }

        binding.btnRemoveImage.setOnClickListener {
            viewModel.clearImage()
        }
    }

    // ─── State Observation ─────────────────────────────────────

    private fun observeUiState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect { state ->
                    // Update status text
                    binding.tvStatus.text = state.statusMessage

                    // Update status text color based on engine status
                    val statusColor = when (state.engineStatus) {
                        EngineStatus.READY -> R.color.status_success
                        EngineStatus.ERROR -> R.color.status_error
                        EngineStatus.DOWNLOADING, EngineStatus.INITIALIZING -> R.color.status_warning
                        EngineStatus.GENERATING -> R.color.accent_primary
                        else -> R.color.text_tertiary
                    }
                    binding.tvStatus.setTextColor(ContextCompat.getColor(this@MainActivity, statusColor))

                    // Update backend chips
                    updateBackendChips(state.selectedBackend)

                    // Update chat messages
                    chatAdapter.submitList(state.messages) {
                        // Scroll to bottom after list update
                        if (state.messages.isNotEmpty()) {
                            binding.chatRecyclerView.smoothScrollToPosition(state.messages.size - 1)
                        }
                    }

                    // Show/hide welcome overlay
                    binding.welcomeOverlay.visibility =
                        if (state.messages.isEmpty() && state.engineStatus != EngineStatus.DOWNLOADING)
                            View.VISIBLE else View.GONE

                    // Show/hide loading overlay
                    when (state.engineStatus) {
                        EngineStatus.DOWNLOADING -> {
                            binding.loadingOverlay.visibility = View.VISIBLE
                            binding.progressBar.visibility = View.VISIBLE
                            binding.progressBar.progress = (state.downloadProgress * 1000).toInt()
                            binding.tvProgressPercent.text = "${(state.downloadProgress * 100).toInt()}%"
                            binding.tvLoadingTitle.text = "Downloading Model"
                            binding.tvLoadingSubtitle.text = "FastVLM-0.5B for ${state.selectedBackend}"
                        }
                        EngineStatus.INITIALIZING -> {
                            binding.loadingOverlay.visibility = View.VISIBLE
                            binding.progressBar.visibility = View.GONE
                            binding.tvProgressPercent.text = ""
                            binding.tvLoadingTitle.text = "Initializing Engine"
                            binding.tvLoadingSubtitle.text = "Setting up ${state.selectedBackend} backend..."
                        }
                        else -> {
                            binding.loadingOverlay.visibility = View.GONE
                        }
                    }

                    // Image preview
                    if (state.currentImageUri != null) {
                        binding.imagePreviewContainer.visibility = View.VISIBLE
                        binding.ivPreview.load(state.currentImageUri) {
                            crossfade(true)
                            transformations(RoundedCornersTransformation(32f))
                        }
                    } else {
                        binding.imagePreviewContainer.visibility = View.GONE
                    }

                    // Enable/disable send button
                    val canSend = state.engineStatus == EngineStatus.READY
                    binding.btnSend.alpha = if (canSend) 1f else 0.4f
                    binding.btnSend.isEnabled = canSend
                    binding.etMessage.isEnabled = canSend
                }
            }
        }
    }

    // ─── Message Sending ───────────────────────────────────────

    private fun sendMessage() {
        val text = binding.etMessage.text.toString().trim()
        if (text.isEmpty() && viewModel.uiState.value.currentImageUri == null) return

        viewModel.sendMessage(text)
        binding.etMessage.text?.clear()
        hideKeyboard()
    }

    // ─── Image Handling ────────────────────────────────────────

    private fun handleImageSelected(uri: Uri) {
        // Take persistent permission for content URIs
        try {
            contentResolver.takePersistableUriPermission(
                uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (_: SecurityException) {
            // Not all URIs support persistent permissions; that's OK
        }
        viewModel.setImage(uri)
    }

    private fun requestImagePicker() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ uses photo picker, no permission needed
            launchImagePicker()
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED
            ) {
                launchImagePicker()
            } else {
                storagePermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        } else {
            launchImagePicker()
        }
    }

    private fun launchImagePicker() {
        pickMediaLauncher.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
        )
    }

    private fun requestCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            launchCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchCamera() {
        val photoFile = File(cacheDir, "camera_${System.currentTimeMillis()}.jpg")
        cameraImageUri = FileProvider.getUriForFile(
            this,
            "${packageName}.fileprovider",
            photoFile
        )
        takePictureLauncher.launch(cameraImageUri!!)
    }

    // ─── Utility ───────────────────────────────────────────────

    private fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(binding.etMessage.windowToken, 0)
    }
}
