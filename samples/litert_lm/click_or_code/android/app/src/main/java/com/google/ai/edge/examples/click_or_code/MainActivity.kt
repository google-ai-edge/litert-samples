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

package com.google.ai.edge.examples.click_or_code

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
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
 * - `--es bundle <path>`: run this model file (a path in the app's external files dir, or an
 *   absolute path), with no hash check, instead of the catalog's;
 * - `--ez autoload true`: load the model now (from a side-loaded copy, else by downloading it);
 * - `--es example <id>`: show this bundled example, no answer;
 * - `--es image <path>`: show this picture (a path in the app's external files dir), no answer;
 * - `--es request <text>`: the text in the request field;
 * - `--es plan <path>`: run the steps of this plan file (see [AutoRun]) once the model is loaded.
 *
 * A scripted launch shows the screen over the keyguard, turns the display on and keeps it on.
 */
class MainActivity : ComponentActivity() {
  private val vm: StepViewModel by viewModels()

  private val pickPhoto =
    registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
      uri?.let { open(it) }
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    if (scripted(intent)) overKeyguard()
    // A white page: dark status-bar and navigation-bar icons.
    enableEdgeToEdge(
      statusBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
      navigationBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
    )
    setContent {
      val ui by vm.ui.collectAsState()
      // The screen stays on while the model loads or a step runs: a locked phone moves the app's
      // process to the background cpuset (the little cores).
      val keepOn =
        ui.deciding || ui.engine is EngineState.Preparing || ui.model is ModelState.Downloading
      LaunchedEffect(keepOn) {
        if (keepOn) {
          window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else if (!scriptedMode) {
          window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
      }
      StepScreen(
        ui,
        onPick = {
          pickPhoto.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
          )
        },
        onExample = { showExample(it) },
        onRequest = vm::setRequest,
        onDecide = vm::decide,
        onStop = vm::stop,
        onDownload = vm::download,
        onCopy = { text -> copy(text) },
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

  private var scriptedMode = false

  private fun showExample(example: Example) {
    val bytes = assets.open("examples/${example.file}").use { it.readBytes() }
    vm.setPhoto(bytes, example.name, example.requests)
    vm.setRequest("")
  }

  private fun open(uri: Uri) {
    val name =
      contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c
        ->
        if (c.moveToFirst()) c.getString(0) else null
      } ?: "screenshot"
    val bytes =
      runCatching { contentResolver.openInputStream(uri)?.use { it.readBytes() } }.getOrNull()
    if (bytes == null) {
      Toast.makeText(this, "Cannot read that picture", Toast.LENGTH_SHORT).show()
      return
    }
    vm.setPhoto(bytes, name)
    vm.setRequest("")
  }

  private fun copy(text: String) {
    getSystemService(ClipboardManager::class.java)
      .setPrimaryClip(ClipData.newPlainText("Command", text))
    Log.i(StepViewModel.TAG, "COPIED chars=${text.length}")
    // Android 13 and later show their own confirmation of a copy.
    if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.S_V2) {
      Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show()
    }
  }

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
      StepViewModel.TAG,
      "extras bundle=${i.getStringExtra("bundle")} autoload=${i.getBooleanExtra("autoload", false)} " +
        "example=${i.getStringExtra("example")} image=${i.getStringExtra("image")} " +
        "plan=${i.getStringExtra("plan")}",
    )
    i.getStringExtra("bundle")?.let { path ->
      val f = if (path.startsWith("/")) File(path) else File(root, path)
      if (f.isFile) vm.useBundle(f) else Log.w(StepViewModel.TAG, "no bundle at ${f.path}")
    }
    if (!i.hasExtra("bundle") && (i.getBooleanExtra("autoload", false) || i.hasExtra("plan"))) {
      vm.download()
    }
    i.getStringExtra("example")?.let { id ->
      vm.ui.value.examples.firstOrNull { it.id == id }?.let { showExample(it) }
        ?: Log.w(StepViewModel.TAG, "no example $id")
    }
    i.getStringExtra("image")?.let { path ->
      val f = File(root, path)
      if (f.isFile) vm.setPhoto(f.readBytes(), f.name)
      else Log.w(StepViewModel.TAG, "no image at ${f.path}")
    }
    i.getStringExtra("request")?.let { vm.setRequest(it) }
    i.getStringExtra("plan")?.let { AutoRun(vm, root, assets).start(lifecycleScope, it) }
  }

  private companion object {
    val SCRIPT_EXTRAS = listOf("bundle", "autoload", "example", "image", "request", "plan")
  }
}
