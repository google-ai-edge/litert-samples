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

package com.google.ai.edge.examples.voice_to_text

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.lifecycleScope
import java.io.File

/**
 * The app's one screen. Launch extras, for scripted runs in a debuggable build (a release build
 * ignores them; all optional; a running screen takes them again):
 * - `--ez autoload true`: load the model now (from a side-loaded copy, else by downloading it);
 * - `--es plan <path>`: once the model is loaded, run the WAV files of this plan (see [AutoRun]);
 * - `--ez save_audio true`: write each recording and its text under `runs/` in the external files dir;
 * - `--ei max_seconds <n>`: the recording cap (1 to 90 s);
 * - `--el need_bytes <n>`: the free memory the check asks for before the Engine is made;
 * - `--es task TRANSCRIPT|ENGLISH`, `--es keywords <text>`: the task and the keyword field;
 * - `--ez clear true`: clear the transcripts and the message.
 *
 * A scripted launch shows the screen over the keyguard, turns the display on and keeps it on.
 */
class MainActivity : ComponentActivity() {
  private val vm: VoiceViewModel by viewModels()

  private val askMicrophone =
    registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) {
        Log.i(VoiceViewModel.TAG, "PERMISSION_GRANTED")
        vm.record()
      } else {
        // After a second refusal the system stops showing the dialog and the rationale flag is
        // false: only the app's settings page can turn the microphone on.
        vm.permissionRefused(blocked = !shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO))
      }
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    if (scripted(intent)) overKeyguard()
    enableEdgeToEdge()
    setContent {
      val ui by vm.ui.collectAsState()
      // The screen stays on while the model loads, a recording runs or text is written: a locked
      // phone moves the app's process to the background cpuset (the little cores).
      val keepOn =
        ui.recording ||
          ui.transcribing ||
          ui.engine is EngineState.Loading ||
          ui.engine is EngineState.WarmingUp ||
          ui.model is ModelState.Downloading
      LaunchedEffect(keepOn) {
        if (keepOn) {
          window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else if (!scriptedMode) {
          window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
      }
      VoiceScreen(
        ui,
        onDownload = vm::download,
        onLoad = vm::loadAgain,
        onTask = vm::setTask,
        onKeywords = vm::setKeywords,
        onRecord = { record() },
        onStop = vm::stopRecording,
        onCancel = vm::cancel,
        onAgain = vm::again,
        onCopy = { text -> copy(text) },
        onMessageAction = { action ->
          if (action == MessageAction.OPEN_SETTINGS) openAppSettings()
        },
        onDismissMessage = vm::dismissMessage,
      )
    }
    if (savedInstanceState == null) handle(intent)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    if (scripted(intent)) overKeyguard()
    handle(intent)
  }

  override fun onStop() {
    super.onStop()
    // The microphone is the app's only while it is on screen.
    if (!isChangingConfigurations) vm.onBackground()
  }

  private fun record() {
    vm.recordBlocker()?.let {
      // No model yet: say so before asking for the microphone.
      vm.record()
      return
    }
    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
      vm.record()
    } else {
      Log.i(VoiceViewModel.TAG, "PERMISSION_ASK")
      askMicrophone.launch(Manifest.permission.RECORD_AUDIO)
    }
  }

  private fun openAppSettings() {
    Log.i(VoiceViewModel.TAG, "OPEN_SETTINGS")
    vm.dismissMessage()
    startActivity(
      Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null))
        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    )
  }

  private fun copy(text: String) {
    getSystemService(ClipboardManager::class.java)
      .setPrimaryClip(ClipData.newPlainText("Transcript", text))
    Log.i(VoiceViewModel.TAG, "COPIED chars=${text.length}")
    // Android 13 and later show their own confirmation of a copy.
    if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.S_V2) {
      Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show()
    }
  }

  private var scriptedMode = false

  /** The script extras are for development: a release build ignores them. */
  private fun debuggable(): Boolean =
    (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0

  private fun scripted(i: Intent) = debuggable() && SCRIPT_EXTRAS.any { i.hasExtra(it) }

  private fun overKeyguard() {
    setShowWhenLocked(true)
    setTurnScreenOn(true)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    scriptedMode = true
  }

  private fun handle(i: Intent) {
    if (!debuggable()) return
    val root = getExternalFilesDir(null) ?: return
    Log.i(
      VoiceViewModel.TAG,
      "extras autoload=${i.getBooleanExtra("autoload", false)} plan=${i.getStringExtra("plan")} " +
        "save_audio=${i.getBooleanExtra("save_audio", false)} max_seconds=${i.getIntExtra("max_seconds", 0)} " +
        "need_bytes=${i.getLongExtra("need_bytes", 0)} task=${i.getStringExtra("task")} " +
        "clear=${i.getBooleanExtra("clear", false)}",
    )
    if (i.getBooleanExtra("clear", false)) vm.clearResults()
    if (i.hasExtra("save_audio")) {
      vm.saveDir = if (i.getBooleanExtra("save_audio", false)) File(root, "runs") else null
    }
    if (i.hasExtra("max_seconds")) vm.setMaxSeconds(i.getIntExtra("max_seconds", Prompts.MAX_SECONDS))
    i.getStringExtra("task")?.let { t -> runCatching { vm.setTask(Task.valueOf(t)) } }
    i.getStringExtra("keywords")?.let { vm.setKeywords(it) }
    if (i.hasExtra("need_bytes")) vm.setNeededMemory(i.getLongExtra("need_bytes", MemoryCheck.NEEDED_BYTES))
    if (i.getBooleanExtra("autoload", false) || i.hasExtra("plan")) vm.download()
    i.getStringExtra("plan")?.let { AutoRun(vm, root).start(lifecycleScope, it) }
  }

  private companion object {
    val SCRIPT_EXTRAS =
      listOf("autoload", "plan", "save_audio", "max_seconds", "need_bytes", "task", "keywords", "clear")
  }
}
