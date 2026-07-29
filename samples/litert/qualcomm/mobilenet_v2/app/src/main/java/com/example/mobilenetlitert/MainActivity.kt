package com.example.mobilenetlitert

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContent { MobileNetApp() }
  }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MobileNetApp() {
  val context = LocalContext.current
  val scope = rememberCoroutineScope()
  val classifier = remember { MobilenetClassifier(context.applicationContext) }
  val policy = remember { DevicePolicy.defaultBackendPolicy() }

  var bitmap by remember { mutableStateOf<Bitmap?>(null) }
  var result by remember { mutableStateOf<ClassificationResult?>(null) }
  var status by remember { mutableStateOf("Starting") }
  var error by remember { mutableStateOf<String?>(null) }
  var busy by remember { mutableStateOf(false) }

  val imagePicker =
    rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
      uri ?: return@rememberLauncherForActivityResult
      runCatching {
        val stream = requireNotNull(context.contentResolver.openInputStream(uri)) {
          "Unable to open selected image."
        }
        stream.use { input ->
          requireNotNull(BitmapFactory.decodeStream(input)) { "Unable to decode selected image." }
        }
      }.onSuccess {
        bitmap = it
        result = null
        error = null
      }.onFailure {
        error = it.message
        Log.e(MobilenetClassifier.TAG, "Image selection failed", it)
      }
    }

  LaunchedEffect(Unit) {
    bitmap = loadSampleBitmap(context)
    busy = true
    status = "Initializing ${if (policy == BackendPolicy.CPU_EMULATOR) "CPU" else "NPU"}"
    runCatching { classifier.initialize(policy) }
      .onSuccess {
        status = if (policy == BackendPolicy.CPU_EMULATOR) "CPU ready" else "NPU ready"
      }
      .onFailure {
        error = it.message
        status = "Initialization failed"
        Log.e(MobilenetClassifier.TAG, "Initialization failed", it)
      }
    busy = false
  }

  DisposableEffect(Unit) {
    onDispose { classifier.close() }
  }

  MaterialTheme(
    colorScheme =
      MaterialTheme.colorScheme.copy(
        primary = Color(0xFF0F766E),
        secondary = Color(0xFFB45309),
        surface = Color(0xFFF7F8FA),
        background = Color(0xFFF7F8FA),
      )
  ) {
    Scaffold(
      topBar = { TopAppBar(title = { Text("LiteRT MobileNet") }) },
      containerColor = Color(0xFFF7F8FA),
    ) { padding ->
      Column(
        modifier =
          Modifier
            .padding(padding)
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
      ) {
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically,
        ) {
          BackendBadge(status = status, policy = policy)
          if (busy) CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 3.dp)
        }

        bitmap?.let {
          Image(
            bitmap = it.asImageBitmap(),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier =
              Modifier
                .fillMaxWidth()
                .height(300.dp)
                .clip(RoundedCornerShape(8.dp)),
          )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
          Button(onClick = { imagePicker.launch("image/*") }, enabled = !busy) {
            Icon(Icons.Outlined.PhotoLibrary, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Image")
          }
          Button(
            onClick = {
              val currentBitmap = bitmap ?: return@Button
              scope.launch {
                busy = true
                error = null
                status = "Running"
                runCatching { classifier.classify(currentBitmap) }
                  .onSuccess {
                    result = it
                    status = "${it.backend} complete"
                  }
                  .onFailure {
                    error = it.message
                    status = "Run failed"
                    Log.e(MobilenetClassifier.TAG, "Inference failed", it)
                  }
                busy = false
              }
            },
            enabled = !busy && bitmap != null && error == null,
          ) {
            Icon(Icons.Outlined.PlayArrow, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Run")
          }
        }

        error?.let { ErrorPanel(it) }
        result?.let { ResultPanel(it) }
      }
    }
  }
}

@Composable
private fun BackendBadge(status: String, policy: BackendPolicy) {
  val color = if (policy == BackendPolicy.CPU_EMULATOR) Color(0xFF1D4ED8) else Color(0xFF0F766E)
  Box(
    modifier =
      Modifier
        .clip(RoundedCornerShape(8.dp))
        .background(color.copy(alpha = 0.12f))
        .padding(horizontal = 12.dp, vertical = 8.dp)
  ) {
    Text(status, color = color, fontWeight = FontWeight.SemiBold)
  }
}

@Composable
private fun ErrorPanel(message: String) {
  Surface(color = Color(0xFFFFE4E6), shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth()) {
    Text(
      text = message,
      color = Color(0xFF9F1239),
      modifier = Modifier.padding(14.dp),
      style = MaterialTheme.typography.bodyMedium,
    )
  }
}

@Composable
private fun ResultPanel(result: ClassificationResult) {
  Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
    Text(
      "${result.backend}  inference ${result.inferenceMs} ms  preprocess ${result.preprocessMs} ms",
      style = MaterialTheme.typography.bodyMedium,
      color = Color(0xFF334155),
    )

    result.predictions.forEach { classification ->
      Column(verticalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxWidth()) {
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically,
        ) {
          Text(classification.label, fontWeight = FontWeight.SemiBold)
          Text("${(classification.score * 100).formatOne()}%")
        }
        LinearProgressIndicator(
          progress = { classification.score.coerceIn(0.0f, 1.0f) },
          modifier = Modifier.fillMaxWidth(),
        )
      }
    }
  }
}

private fun Float.formatOne(): String = String.format("%.1f", this)

private fun loadSampleBitmap(context: android.content.Context): Bitmap {
  return runCatching {
    context.assets.open("sample/grace_hopper.jpg").use { BitmapFactory.decodeStream(it) }
  }.getOrNull() ?: Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888).apply {
    eraseColor(android.graphics.Color.rgb(32, 129, 120))
  }
}
