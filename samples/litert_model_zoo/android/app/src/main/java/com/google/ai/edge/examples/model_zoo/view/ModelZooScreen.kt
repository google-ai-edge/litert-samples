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

package com.google.ai.edge.examples.model_zoo.view

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.SystemClock
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.google.ai.edge.examples.model_zoo.BuildConfig
import com.google.ai.edge.examples.model_zoo.MainViewModel
import com.google.ai.edge.examples.model_zoo.R
import com.google.ai.edge.examples.model_zoo.ResultCounts
import com.google.ai.edge.examples.model_zoo.TaskFailures
import com.google.ai.edge.examples.model_zoo.UiState
import com.google.ai.edge.examples.model_zoo.audio.BatchAudioTasks
import com.google.ai.edge.examples.model_zoo.common.RealtimeCameraPipeline
import com.google.ai.edge.examples.model_zoo.data.DownloadState
import com.google.ai.edge.examples.model_zoo.data.DownloadStatus
import com.google.ai.edge.examples.model_zoo.data.ModelEntry
import com.google.ai.edge.examples.model_zoo.data.OpenSourceCredits
import com.google.ai.edge.examples.model_zoo.image.RealtimeImageTasks
import com.google.ai.edge.examples.model_zoo.image.SingleImageTasks

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelZooScreen(state: UiState, vm: MainViewModel, activity: ComponentActivity) {
  val selected = state.tasks.firstOrNull { it.taskId == state.selectedTaskId }
  BackHandler(state.screen == "task" || state.screen == "licenses") {
    vm.navigate(if (state.screen == "licenses") "about" else "home")
  }
  state.downloadConfirmation?.let { confirmation ->
    AlertDialog(
      onDismissRequest = vm::cancelDownloadConfirmation,
      title = { Text(stringResource(R.string.download_confirm_title, confirmation.model)) },
      text = {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
          Text(
            stringResource(
              R.string.download_confirm_size,
              sizeLabel(confirmation.bytes),
              confirmation.bytes,
            )
          )
          Text(stringResource(R.string.download_confirm_storage))
          if (confirmation.metered)
            Text(
              stringResource(R.string.download_metered_warning),
              color = MaterialTheme.colorScheme.error,
            )
        }
      },
      confirmButton = {
        Button(onClick = vm::confirmDownload) {
          Text(stringResource(R.string.download_confirm_action))
        }
      },
      dismissButton = {
        TextButton(onClick = vm::cancelDownloadConfirmation) {
          Text(stringResource(R.string.cancel))
        }
      },
    )
  }
  Scaffold(
    modifier = Modifier.fillMaxSize(),
    topBar = {
      TopAppBar(
        title = {
          Text(
            when (state.screen) {
              "licenses" -> stringResource(R.string.open_source_licenses)
              else -> selected?.task ?: stringResource(R.string.app_name)
            },
            style = MaterialTheme.typography.titleLarge,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
          )
        },
        colors =
          TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
        navigationIcon = {
          if (state.screen == "task" || state.screen == "licenses") {
            TextButton(
              onClick = { vm.navigate(if (state.screen == "licenses") "about" else "home") }
            ) {
              Text(stringResource(R.string.back))
            }
          }
        },
      )
    },
    bottomBar = {
      NavigationBar(containerColor = MaterialTheme.colorScheme.surface, tonalElevation = 0.dp) {
        listOf("home" to R.string.home, "models" to R.string.models, "about" to R.string.about)
          .forEach { (screen, label) ->
            val active =
              state.screen == screen ||
                (screen == "about" && state.screen == "licenses")
            NavigationBarItem(
              selected = active,
              onClick = { vm.navigate(screen) },
              icon = { NavigationMark(screen, stringResource(label), active) },
              label = { Text(stringResource(label)) },
            )
          }
      }
    },
  ) { padding ->
    Column(Modifier.fillMaxSize().padding(padding)) {
      state.error?.let { error ->
        Card(
          Modifier.fillMaxWidth().padding(horizontal = 16.dp),
          colors =
            CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        ) {
          Column(Modifier.padding(12.dp)) {
            Text(error, style = MaterialTheme.typography.bodyMedium)
            TextButton(onClick = vm::clearError) { Text(stringResource(R.string.dismiss)) }
          }
        }
      }
      if (state.loading) {
        Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
          CircularProgressIndicator()
          Text(stringResource(R.string.loading))
        }
      } else
        when (state.screen) {
          "models" -> ModelsScreen(state, vm)
          "about" -> AboutScreen(state, vm)
          "licenses" -> OpenSourceLicensesScreen(vm)
          "task" -> selected?.let { TaskScreen(it, state, vm, activity) }
          else -> HomeScreen(state, vm)
        }
    }
  }
}

@Composable
private fun NavigationMark(screen: String, description: String, active: Boolean) {
  val color =
    if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
  Canvas(Modifier.size(24.dp).semantics { contentDescription = description }) {
    val stroke = 1.8.dp.toPx()
    when (screen) {
      "home" -> {
        val side = size.width * 0.3f
        for (x in listOf(0.1f, 0.6f)) for (y in listOf(0.1f, 0.6f)) drawRoundRect(
          color,
          Offset(size.width * x, size.height * y),
          Size(side, side),
          androidx.compose.ui.geometry.CornerRadius(2.dp.toPx()),
          style = Stroke(stroke),
        )
      }
      "models" -> {
        for (y in listOf(0.2f, 0.5f, 0.8f)) drawLine(
          color,
          Offset(size.width * 0.15f, size.height * y),
          Offset(size.width * 0.85f, size.height * y),
          stroke,
        )
      }
      else -> {
        drawCircle(color, size.width * 0.4f, style = Stroke(stroke))
        drawCircle(color, stroke * 0.65f, Offset(size.width * 0.5f, size.height * 0.3f))
        drawLine(
          color,
          Offset(size.width * 0.5f, size.height * 0.48f),
          Offset(size.width * 0.5f, size.height * 0.73f),
          stroke,
        )
      }
    }
  }
}

@Composable
private fun ModelsScreen(state: UiState, vm: MainViewModel) {
  LazyColumn(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
    item {
      Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
          stringResource(R.string.storage_used, sizeLabel(state.storageBytes)),
          style = MaterialTheme.typography.headlineSmall,
        )
        Text(stringResource(R.string.model_delivery), style = MaterialTheme.typography.bodyMedium)
      }
    }
    items(state.tasks, key = { it.taskId }) { entry ->
      Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
          Text(entry.task, style = MaterialTheme.typography.titleMedium)
          Text(entry.model, style = MaterialTheme.typography.bodyMedium)
          ModelControls(entry, state, vm)
        }
      }
    }
    item { Spacer(Modifier.height(16.dp)) }
  }
}

@Composable
private fun ModelControls(
  entry: ModelEntry,
  state: UiState,
  vm: MainViewModel,
  showFiles: Boolean = false,
) {
  val download = state.downloads[entry.taskId] ?: DownloadState(totalBytes = entry.totalBytes)
  val downloading =
    download.status in
      setOf(DownloadStatus.STARTING, DownloadStatus.DOWNLOADING, DownloadStatus.VERIFYING)
  Text(
    "${sizeLabel(entry.totalBytes)} · ${entryState(download)}",
    style = MaterialTheme.typography.bodySmall,
    color = MaterialTheme.colorScheme.primary,
  )
  if (showFiles) {
    Text("${entry.totalBytes} bytes total", style = MaterialTheme.typography.bodySmall)
    entry.files.forEach {
      Text("${it.name} · ${it.bytes} bytes", style = MaterialTheme.typography.bodySmall)
    }
  }
  if (downloading) {
    if (download.status == DownloadStatus.STARTING || download.status == DownloadStatus.VERIFYING) {
      LinearProgressIndicator(Modifier.fillMaxWidth())
    } else {
      LinearProgressIndicator(
        progress = {
          if (entry.totalBytes > 0)
            (download.receivedBytes.toDouble() / entry.totalBytes).toFloat().coerceIn(0f, 1f)
          else 0f
        },
        modifier = Modifier.fillMaxWidth(),
      )
    }
    Text(
      stringResource(
        R.string.download_progress,
        sizeLabel(download.receivedBytes),
        sizeLabel(entry.totalBytes),
      ),
      style = MaterialTheme.typography.bodySmall,
    )
  }
  download.error?.let {
    Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
  }
  if (entry.canDownload) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      if (downloading) {
        OutlinedButton(onClick = { vm.pauseDownload(entry.taskId) }) {
          Text(stringResource(R.string.pause))
        }
      } else if (download.status != DownloadStatus.READY) {
        Button(onClick = { vm.download(entry.taskId) }) {
          Text(
            stringResource(
              when (download.status) {
                DownloadStatus.PAUSED -> R.string.resume
                DownloadStatus.ERROR -> R.string.retry
                else -> R.string.download
              }
            )
          )
        }
      }
      if (download.receivedBytes > 0 || download.status == DownloadStatus.READY) {
        OutlinedButton(
          onClick = { vm.delete(entry.taskId) },
          enabled = !state.busy && !state.camera && !state.recording && !downloading,
        ) {
          Text(stringResource(R.string.delete))
        }
      }
    }
  } else
    Text(
      stringResource(R.string.verification_pending),
      style = MaterialTheme.typography.bodyMedium,
    )
}

@Composable
internal fun DetailsSection(content: @Composable () -> Unit) {
  var expanded by remember { mutableStateOf(false) }
  TextButton(onClick = { expanded = !expanded }) {
    Text(if (expanded) "Hide details" else "Details")
  }
  if (expanded) Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { content() }
}

@Composable
private fun TaskScreen(
  entry: ModelEntry,
  state: UiState,
  vm: MainViewModel,
  activity: ComponentActivity,
) {
  val context = LocalContext.current
  val navigationTapGuard = remember(entry.taskId) { NavigationTapGuard(SystemClock.uptimeMillis()) }
  val permissionHistory =
    remember(context) {
      context.applicationContext.getSharedPreferences("permission-denials", Context.MODE_PRIVATE)
    }
  var permissionRationale by remember { mutableStateOf<String?>(null) }
  var permissionSettings by remember { mutableStateOf<String?>(null) }
  var pendingSettingsPermission by remember { mutableStateOf<String?>(null) }
  fun isGranted(permission: String): Boolean =
    ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
  fun deniedMessage(permission: String): String =
    context.getString(
      if (permission == Manifest.permission.CAMERA) R.string.camera_permission_denied
      else if (entry.taskId == "speech-recognition") R.string.microphone_permission_denied
      else R.string.microphone_audio_permission_denied
    )
  fun denied(permission: String) {
    permissionHistory.edit().putBoolean(permission, true).apply()
    if (
      permissionDecision(
        granted = false,
        previouslyDenied = true,
        shouldShowRationale = activity.shouldShowRequestPermissionRationale(permission),
      ) == PermissionDecision.OPEN_SETTINGS
    ) {
      vm.clearError()
      permissionSettings = permission
    } else vm.showError(deniedMessage(permission))
  }
  val imagePicker =
    rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
      uri?.let(vm::pickImage)
    }
  val photo =
    rememberLauncherForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
      bitmap?.let(vm::photograph)
    }
  val secondImagePicker =
    rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
      uri?.let(vm::pickSecondImage)
    }
  var pendingPhoto by remember { mutableStateOf(false) }
  fun continuePermissionAction(permission: String) {
    // A later revoke with cleared Android flags must once again be a first-time request.
    permissionHistory.edit().remove(permission).apply()
    vm.clearError()
    if (permission == Manifest.permission.CAMERA) {
      if (pendingPhoto) photo.launch(null) else vm.setCamera(true)
    } else vm.startRecording()
  }
  val cameraPermission =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) continuePermissionAction(Manifest.permission.CAMERA)
      else denied(Manifest.permission.CAMERA)
    }
  if (state.camera && entry.taskId in (RealtimeImageTasks.ids + "object-detection"))
    CameraSession(activity, vm)
  val ready = state.downloads[entry.taskId]?.status == DownloadStatus.READY
  val wavPicker =
    rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
      uri?.let(vm::pickWav)
    }
  val speechSaveContract = remember {
    object : ActivityResultContracts.CreateDocument("audio/wav") {
      override fun createIntent(context: Context, input: String): Intent =
        super.createIntent(context, input).putExtra(Intent.EXTRA_LOCAL_ONLY, true)
    }
  }
  val speechSaver =
    rememberLauncherForActivityResult(speechSaveContract) { uri -> uri?.let(vm::saveSpeech) }
  val microphonePermission =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) continuePermissionAction(Manifest.permission.RECORD_AUDIO)
      else denied(Manifest.permission.RECORD_AUDIO)
    }
  val settingsLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
      val permission = pendingSettingsPermission
      pendingSettingsPermission = null
      if (permission != null) {
        if (isGranted(permission)) continuePermissionAction(permission)
        else vm.showError(deniedMessage(permission))
      }
    }
  fun requestPermission(permission: String) {
    when (
      permissionDecision(
        granted = isGranted(permission),
        previouslyDenied = permissionHistory.getBoolean(permission, false),
        shouldShowRationale = activity.shouldShowRequestPermissionRationale(permission),
      )
    ) {
      PermissionDecision.PROCEED -> continuePermissionAction(permission)
      PermissionDecision.SHOW_RATIONALE -> permissionRationale = permission
      PermissionDecision.OPEN_SETTINGS -> permissionSettings = permission
    }
  }
  fun requestCamera() = requestPermission(Manifest.permission.CAMERA)
  fun requestMicrophone() = requestPermission(Manifest.permission.RECORD_AUDIO)
  permissionRationale?.let { permission ->
    AlertDialog(
      onDismissRequest = { permissionRationale = null },
      title = {
        Text(
          stringResource(
            if (permission == Manifest.permission.CAMERA) R.string.camera_rationale_title
            else R.string.microphone_rationale_title
          )
        )
      },
      text = {
        Text(
          stringResource(
            if (permission == Manifest.permission.CAMERA) R.string.camera_rationale
            else if (entry.taskId == "speech-recognition") R.string.microphone_rationale
            else R.string.microphone_audio_rationale
          )
        )
      },
      confirmButton = {
        TextButton(
          onClick = {
            permissionRationale = null
            if (permission == Manifest.permission.CAMERA) cameraPermission.launch(permission)
            else microphonePermission.launch(permission)
          }
        ) {
          Text(stringResource(R.string.permission_continue))
        }
      },
      dismissButton = {
        TextButton(onClick = { permissionRationale = null }) {
          Text(stringResource(R.string.cancel))
        }
      },
    )
  }
  permissionSettings?.let { permission ->
    AlertDialog(
      onDismissRequest = { permissionSettings = null },
      title = { Text(stringResource(R.string.permission_settings_title)) },
      text = {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
          Text(deniedMessage(permission))
          Text(stringResource(R.string.permission_settings_message))
        }
      },
      confirmButton = {
        TextButton(
          onClick = {
            permissionSettings = null
            pendingSettingsPermission = permission
            settingsLauncher.launch(
              Intent(
                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.fromParts("package", context.packageName, null),
              )
            )
          }
        ) {
          Text(stringResource(R.string.open_settings))
        }
      },
      dismissButton = {
        TextButton(onClick = { permissionSettings = null }) {
          Text(stringResource(R.string.cancel))
        }
      },
    )
  }

  val listState = rememberLazyListState()
  LaunchedEffect(state.selectedTaskId) { listState.scrollToItem(0) }
  LaunchedEffect(state.inputImage, state.secondaryImage) {
    if (state.inputImage != null && state.inferenceMs == null && !state.camera)
      listState.animateScrollToItem(0)
  }
  LaunchedEffect(state.inferenceMs, state.busy) {
    if (state.inferenceMs != null && !state.busy && !state.camera) listState.animateScrollToItem(2)
  }
  BoxWithConstraints(Modifier.fillMaxSize().guardNavigationTaps(navigationTapGuard)) {
    // Reserve the rest of the viewport outside the natural-size result card. This lets the
    // list align a short result at the top instead of leaving clipped input controls above it.
    val resultMinHeight = (maxHeight - 32.dp).coerceAtLeast(0.dp)
    LazyColumn(
      Modifier.fillMaxSize(),
      state = listState,
      contentPadding = PaddingValues(16.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      item(key = "model-header") {
        Column {
          Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
              entry.model,
              Modifier.weight(1f),
              style = MaterialTheme.typography.labelLarge,
              maxLines = 1,
              overflow = TextOverflow.Ellipsis,
            )
            Text(
              "· ${sizeLabel(entry.totalBytes)} · ${entryState(state.downloads[entry.taskId])}",
              style = MaterialTheme.typography.labelLarge,
              maxLines = 1,
            )
          }
          DetailsSection {
            Text(entry.model, style = MaterialTheme.typography.titleSmall)
            ModelControls(entry, state, vm, showFiles = true)
            if (entry.taskId == "video-action-recognition")
              Text(
                stringResource(R.string.movinet_cpu_note),
                style = MaterialTheme.typography.bodySmall,
              )
          }
          if (!ready) ModelControls(entry, state, vm)
        }
      }
      item(key = "task-input") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
          if (!ready) {
            Text(stringResource(R.string.download_first))
          } else if (entry.taskId == "object-detection") {
            PhotoButtons(
              choosePhoto = {
                imagePicker.launch(
                  PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
              },
              takePhoto = {
                pendingPhoto = true
                requestCamera()
              },
              enabled = !state.busy && !state.camera,
            )
            if (state.camera) DetectionImage(state, compact = true)
            else
              state.inputImage?.let {
                ImagePreview(it, stringResource(R.string.input_image), compact = true)
              }
            if (!state.camera)
              Button(
                onClick = vm::runDetectionImage,
                modifier = Modifier.fillMaxWidth(),
                enabled = !state.busy && state.inputImage != null,
              ) {
                Text(stringResource(R.string.run_image))
              }
            OutlinedButton(
              onClick = {
                if (state.camera) vm.setCamera(false)
                else {
                  pendingPhoto = false
                  requestCamera()
                }
              },
              enabled = !state.busy,
              modifier = Modifier.fillMaxWidth(),
            ) {
              Text(
                stringResource(if (state.camera) R.string.stop_camera else R.string.start_camera)
              )
            }
          } else if (entry.taskId in SingleImageTasks.ids) {
            if (state.camera) {
              state.outputImage?.let {
                ImagePreview(it, "Live camera with model overlay", compact = true)
              }
              Text(
                "Live camera · ${state.cameraFrames} frames processed",
                style = MaterialTheme.typography.bodySmall,
              )
            } else
              SingleImagePanel(
                entry,
                state,
                vm,
                choosePhoto = {
                  imagePicker.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                  )
                },
                takePhoto = {
                  pendingPhoto = true
                  requestCamera()
                },
                chooseSecondPhoto = {
                  secondImagePicker.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                  )
                },
              )
            if (entry.taskId in RealtimeImageTasks.ids) {
              OutlinedButton(
                onClick = {
                  if (state.camera) vm.setCamera(false)
                  else {
                    pendingPhoto = false
                    requestCamera()
                  }
                },
                enabled = !state.busy,
                modifier = Modifier.fillMaxWidth(),
              ) {
                Text(
                  stringResource(if (state.camera) R.string.stop_camera else R.string.start_camera)
                )
              }
              if (entry.taskId in setOf("head-pose-estimation", "face-liveness-anti-spoofing"))
                Text("Center a face in the frame.", style = MaterialTheme.typography.bodySmall)
            }
          } else if (entry.taskId in BatchAudioTasks.ids) {
            if (entry.taskId == "audio-source-separation")
              Text(
                stringResource(R.string.tiger_duration_hint),
                style = MaterialTheme.typography.bodyMedium,
              )
            BatchAudioPanel(
              state,
              vm,
              record = { requestMicrophone() },
              chooseWav = {
                wavPicker.launch(
                  arrayOf("audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave")
                )
              },
            )
          } else if (entry.taskId == "speech-recognition") {
            Text(stringResource(R.string.asr_hint), style = MaterialTheme.typography.bodyMedium)
            AudioInputLabel(state)
            Button(
              onClick = { if (state.recording) vm.stopRecording() else requestMicrophone() },
              enabled = !state.busy,
              modifier = Modifier.fillMaxWidth(),
            ) {
              Text(
                stringResource(
                  if (state.recording) R.string.stop_recording else R.string.record_audio
                )
              )
            }
            OutlinedButton(
              onClick = {
                wavPicker.launch(
                  arrayOf("audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave")
                )
              },
              enabled = !state.busy && !state.recording,
              modifier = Modifier.fillMaxWidth(),
            ) {
              Text(stringResource(R.string.pick_wav))
            }
            if (state.recording)
              Text(stringResource(R.string.recording_seconds, state.recordedSeconds))
          } else {
            Text(stringResource(R.string.speak_hint), style = MaterialTheme.typography.bodyMedium)
            OutlinedTextField(
              value = state.inputText,
              onValueChange = vm::updateText,
              modifier = Modifier.fillMaxWidth(),
              label = { Text(stringResource(R.string.speak_label)) },
              enabled = !state.busy,
              minLines = 2,
              maxLines = 5,
            )
            Button(
              onClick = vm::synthesize,
              enabled = !state.busy && state.inputText.isNotBlank(),
              modifier = Modifier.fillMaxWidth(),
            ) {
              Text(stringResource(R.string.synthesize))
            }
          }
          if (state.busy) {
            LinearProgressIndicator(Modifier.fillMaxWidth())
            Text(stringResource(R.string.working), style = MaterialTheme.typography.bodySmall)
          }
        }
      }
      item(key = "task-output") {
        if (ready && state.inferenceMs != null && !state.busy) {
          Box(Modifier.fillMaxWidth().heightIn(min = resultMinHeight)) {
            Card(
              Modifier.fillMaxWidth(),
              colors =
                CardDefaults.cardColors(
                  containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                ),
            ) {
              Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                  stringResource(R.string.output_image),
                  style = MaterialTheme.typography.titleMedium,
                )
                when {
                  entry.taskId == "object-detection" -> {
                    DetectionImage(state)
                    Text(ResultCounts.objects(state.boxes.size))
                    state.boxes.take(20).forEach { box ->
                      Text(
                        stringResource(
                          R.string.detection_label,
                          state.labels.getOrElse(box.cls) { box.cls.toString() },
                          box.score * 100,
                        ),
                        style = MaterialTheme.typography.bodyMedium,
                      )
                    }
                  }
                  entry.taskId in SingleImageTasks.ids -> SingleImageResultPanel(entry, state)
                  entry.taskId in BatchAudioTasks.ids -> BatchAudioResultPanel(state, vm)
                  entry.taskId == "speech-recognition" ->
                    SelectionContainer {
                      Text(
                        state.transcript.orEmpty().ifBlank { stringResource(R.string.no_speech) }
                      )
                    }
                  else -> {
                    Text(
                      stringResource(R.string.speech_ready),
                      style = MaterialTheme.typography.bodyMedium,
                    )
                    PlaybackProgress(state)
                    Button(
                      onClick = { if (state.playing) vm.stopPlayback() else vm.playSpeech() },
                      modifier = Modifier.fillMaxWidth(),
                    ) {
                      Text(
                        stringResource(if (state.playing) R.string.stop_playback else R.string.play)
                      )
                    }
                    if (state.speechExportReady)
                      OutlinedButton(
                        onClick = {
                          try {
                            speechSaver.launch("model-zoo-speech.wav")
                          } catch (failure: Throwable) {
                            vm.showError(
                              context.getString(
                                R.string.speech_export_failed,
                                TaskFailures.message(failure),
                              )
                            )
                          }
                        },
                        enabled = !state.speechSaving,
                        modifier = Modifier.fillMaxWidth(),
                      ) {
                        Text(
                          stringResource(
                            if (state.speechSaving) R.string.saving_speech_wav
                            else R.string.save_speech_wav
                          )
                        )
                      }
                    if (state.speechSaved)
                      Text(
                        stringResource(R.string.speech_wav_saved),
                        style = MaterialTheme.typography.bodySmall,
                      )
                  }
                }
                Text(
                  stringResource(R.string.latency, state.inferenceMs, state.backend),
                  style = MaterialTheme.typography.titleMedium,
                  color = MaterialTheme.colorScheme.primary,
                )
                state.fallbackReason?.let {
                  Text(
                    "CPU fallback was used. See Details.",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                  )
                }
                DetailsSection {
                  if (state.backendDetails.isNotBlank())
                    Text(state.backendDetails, style = MaterialTheme.typography.bodySmall)
                  if (state.imageOutputDetails.isNotBlank())
                    SelectionContainer {
                      Text(state.imageOutputDetails, style = MaterialTheme.typography.bodySmall)
                    }
                  if (entry.taskId == "video-action-recognition")
                    Text(
                      stringResource(R.string.movinet_cpu_note),
                      style = MaterialTheme.typography.bodySmall,
                    )
                  state.fallbackReason?.let {
                    Text(
                      stringResource(R.string.fallback, it),
                      color = MaterialTheme.colorScheme.error,
                      style = MaterialTheme.typography.bodySmall,
                    )
                  }
                }
              }
            }
          }
        }
      }
      item { Spacer(Modifier.height(16.dp)) }
    }
  }
}

/** Consume the whole second gesture, including its release, before descendant buttons see it. */
private fun Modifier.guardNavigationTaps(guard: NavigationTapGuard): Modifier =
  pointerInput(guard) {
    awaitPointerEventScope {
      var ignoreGesture = false
      while (true) {
        val event = awaitPointerEvent(PointerEventPass.Initial)
        if (event.changes.any { it.pressed && !it.previousPressed }) {
          ignoreGesture = ignoreGesture || !guard.allows(SystemClock.uptimeMillis())
        }
        if (ignoreGesture) event.changes.forEach { it.consume() }
        if (event.changes.none { it.pressed }) ignoreGesture = false
      }
    }
  }

@Composable
internal fun AudioInputLabel(state: UiState) {
  state.audioInputName?.let { name ->
    Text(name, style = MaterialTheme.typography.titleSmall)
    state.audioInputSeconds?.let {
      Text(
        "%.1f seconds".format(java.util.Locale.ENGLISH, it),
        style = MaterialTheme.typography.bodySmall,
      )
    }
  }
}

@Composable
internal fun PlaybackProgress(state: UiState) {
  val total = state.playbackTotalSeconds
  LinearProgressIndicator(
    progress = { if (total > 0f) (state.playbackElapsedSeconds / total).coerceIn(0f, 1f) else 0f },
    modifier = Modifier.fillMaxWidth(),
  )
  Text(
    "%.1f / %.1f s".format(java.util.Locale.ENGLISH, state.playbackElapsedSeconds, total),
    style = MaterialTheme.typography.bodySmall,
  )
}

@Composable
private fun DetectionImage(state: UiState, compact: Boolean = false) {
  val bitmap = state.image ?: return
  FittedImageFrame(bitmap, compact = compact) {
    Image(
      bitmap.asImageBitmap(),
      stringResource(R.string.image_description),
      Modifier.fillMaxSize(),
      contentScale = ContentScale.Fit,
    )
    Canvas(Modifier.fillMaxSize()) {
      val stroke = 2.5.dp.toPx()
      val textPaint =
        android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
          color = android.graphics.Color.BLACK
          textSize = 13.dp.toPx()
          typeface =
            android.graphics.Typeface.create(
              android.graphics.Typeface.DEFAULT,
              android.graphics.Typeface.BOLD,
            )
        }
      val background =
        android.graphics.Paint().apply { color = android.graphics.Color.rgb(128, 255, 178) }
      state.boxes.forEach { box ->
        val left = ((box.cx - box.w / 2) * size.width).coerceIn(0f, size.width)
        val top = ((box.cy - box.h / 2) * size.height).coerceIn(0f, size.height)
        drawRect(
          Color(0xFF80FFB2),
          Offset(left, top),
          Size(box.w * size.width, box.h * size.height),
          style = Stroke(stroke),
        )
        val label =
          "${state.labels.getOrElse(box.cls) { box.cls.toString() }} ${(box.score * 100).toInt()}%"
        val pad = 4.dp.toPx()
        val labelWidth = (textPaint.measureText(label) + pad * 2).coerceAtMost(size.width)
        val labelHeight = textPaint.fontSpacing + pad
        val x = left.coerceAtMost(size.width - labelWidth)
        val y = (top - labelHeight).coerceAtLeast(0f)
        drawIntoCanvas { canvas ->
          canvas.nativeCanvas.drawRect(x, y, x + labelWidth, y + labelHeight, background)
          canvas.nativeCanvas.drawText(
            label,
            x + pad,
            y + pad / 2 - textPaint.fontMetrics.ascent,
            textPaint,
          )
        }
      }
    }
  }
}

@Composable
private fun CameraSession(activity: ComponentActivity, vm: MainViewModel) {
  DisposableEffect(activity) {
    val pipeline = RealtimeCameraPipeline(activity, onFrame = vm::onCameraFrame)
    var disposed = false
    val future = ProcessCameraProvider.getInstance(activity)
    future.addListener(
      {
        if (!disposed)
          runCatching {
              future.get()
              pipeline.start(activity)
            }
            .onFailure {
              vm.setCamera(false)
              vm.showError(
                activity.getString(R.string.camera_failed, it.message ?: it.javaClass.simpleName)
              )
            }
      },
      ContextCompat.getMainExecutor(activity),
    )
    onDispose {
      disposed = true
      pipeline.enabled = false
      vm.setCamera(false)
      if (future.isDone) runCatching { future.get().unbindAll() }
      pipeline.close()
    }
  }
}

@Composable
private fun AboutScreen(state: UiState, vm: MainViewModel) {
  val uriHandler = LocalUriHandler.current
  val failed = stringResource(R.string.open_link_failed)
  fun open(url: String) {
    runCatching { uriHandler.openUri(url) }.onFailure { vm.showError(failed) }
  }
  LazyColumn(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
    item {
      Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(stringResource(R.string.about_title), style = MaterialTheme.typography.headlineSmall)
        Text(
          "Version ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})",
          style = MaterialTheme.typography.bodyMedium,
        )
        Text(stringResource(R.string.about_description))
        Text(
          stringResource(R.string.powered_by),
          style = MaterialTheme.typography.titleMedium,
          color = MaterialTheme.colorScheme.primary,
        )
        Text(stringResource(R.string.runtime_version), style = MaterialTheme.typography.bodySmall)
        Text(stringResource(R.string.tested_on), style = MaterialTheme.typography.bodyMedium)
        OutlinedButton(onClick = { vm.navigate("licenses") }) {
          Text(stringResource(R.string.open_source_licenses))
        }
      }
    }
    items(state.tasks, key = { it.taskId }) { entry ->
      Card(Modifier.padding(horizontal = 16.dp).fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
          Text(entry.model, style = MaterialTheme.typography.titleMedium)
          Text(
            stringResource(R.string.license, entry.license.name),
            style = MaterialTheme.typography.bodyMedium,
          )
          if (entry.taskId == "audio-classification") {
            Text(OpenSourceCredits.pannsAttribution, style = MaterialTheme.typography.bodyMedium)
            TextButton(onClick = { open(OpenSourceCredits.ccByLicenseUrl) }) {
              Text("CC-BY-4.0 license")
            }
            TextButton(onClick = { open(OpenSourceCredits.pannsSourceUrl) }) {
              Text("PANNs creators and original weights")
            }
            TextButton(onClick = { open(OpenSourceCredits.audioSetSourceUrl) }) {
              Text("AudioSet ontology and attribution")
            }
          }
          Row {
            TextButton(onClick = { open(entry.license.url) }) {
              Text(stringResource(R.string.license_link))
            }
            if (entry.upstream.isNotBlank())
              TextButton(onClick = { open(entry.upstream) }) {
                Text(stringResource(R.string.upstream_link))
              }
          }
          entry.componentLicenses.forEach { component ->
            TextButton(onClick = { open(component.url) }) {
              Text(stringResource(R.string.component_license, component.component, component.name))
            }
          }
          if (entry.modelCard.isNotBlank())
            TextButton(onClick = { open(entry.modelCard) }) {
              Text(stringResource(R.string.model_card_link))
            }
        }
      }
    }
    item { Spacer(Modifier.height(16.dp)) }
  }
}

@Composable
private fun OpenSourceLicensesScreen(vm: MainViewModel) {
  val uriHandler = LocalUriHandler.current
  val failed = stringResource(R.string.open_link_failed)
  fun open(url: String) {
    runCatching { uriHandler.openUri(url) }.onFailure { vm.showError(failed) }
  }
  LazyColumn(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
    item { Text(stringResource(R.string.open_source_intro), Modifier.padding(16.dp)) }
    items(OpenSourceCredits.libraries, key = { it.name }) { library ->
      Card(Modifier.padding(horizontal = 16.dp).fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
          Text(library.name, style = MaterialTheme.typography.titleMedium)
          Text(library.license)
          Row {
            TextButton(onClick = { open(library.licenseUrl) }) {
              Text(stringResource(R.string.license_link))
            }
            TextButton(onClick = { open(library.projectUrl) }) {
              Text(stringResource(R.string.upstream_link))
            }
          }
        }
      }
    }
    item { Spacer(Modifier.height(16.dp)) }
  }
}

@Composable
private fun sizeLabel(bytes: Long): String =
  if (bytes >= 1_000_000_000) stringResource(R.string.size_gb, bytes / 1_000_000_000.0)
  else stringResource(R.string.size_mb, bytes / 1_000_000.0)

@Composable
private fun entryState(state: DownloadState?): String {
  return stringResource(
    when (state?.status ?: DownloadStatus.MISSING) {
      DownloadStatus.MISSING -> R.string.state_missing
      DownloadStatus.STARTING -> R.string.state_starting
      DownloadStatus.DOWNLOADING -> R.string.state_downloading
      DownloadStatus.PAUSED -> R.string.state_paused
      DownloadStatus.VERIFYING -> R.string.state_verifying
      DownloadStatus.READY -> R.string.state_ready
      DownloadStatus.ERROR -> R.string.state_error
    }
  )
}
