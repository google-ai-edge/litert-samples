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

package com.google.ai.edge.examples.math_to_latex

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import java.io.File

/**
 * The app's one screen. Launch extras, for scripted runs in a debuggable build (a release build
 * ignores them; all optional; a running screen takes them again):
 * - `--ez autoload true`: load the model now (from a side-loaded copy, else by downloading it);
 * - `--es image <path>`: show this picture (a path in the app's external files dir), no box, no
 *   answer: the screen a take starts from;
 * - `--es mode EQUATION|TEXT`: the mode;
 * - `--es plan <path>`: run the reads of this plan file (see [AutoRun]) once the model is loaded.
 *
 * A scripted launch shows the screen over the keyguard, turns the display on and keeps it on.
 */
class MainActivity : ComponentActivity() {
  private val vm: MathViewModel by viewModels()

  private val pickPhoto =
    registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
      uri?.let { open(it) }
    }

  /** The camera writes here; the path is fixed so a re-created activity still finds the photo. */
  private val cameraFile by lazy { File(cacheDir, "camera/photo.jpg") }

  private val takePhoto =
    registerForActivityResult(ActivityResultContracts.TakePicture()) { saved ->
      if (saved && cameraFile.isFile) {
        vm.setPhoto(cameraFile.readBytes(), "camera photo")
      }
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    if (scripted(intent)) overKeyguard()
    enableEdgeToEdge()
    setContent {
      val ui by vm.ui.collectAsState()
      // The screen stays on while the model loads or a read runs: a locked phone moves the app's
      // process to the background cpuset (the little cores).
      val keepOn =
        ui.reading || ui.reader is ReaderState.Preparing || ui.model is ModelState.Downloading
      LaunchedEffect(keepOn) {
        if (keepOn) {
          window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else if (!scriptedMode) {
          window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
      }
      MathScreen(
        ui,
        onPickPhoto = {
          pickPhoto.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
          )
        },
        onCamera = {
          cameraFile.parentFile?.mkdirs()
          takePhoto.launch(FileProvider.getUriForFile(this, "$packageName.files", cameraFile))
        },
        onExample = { showExample() },
        onBox = vm::setBox,
        onMode = vm::setMode,
        onRead = vm::read,
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

  private var exampleIndex = 0
  private var scriptedMode = false

  /** The bundled example pages, in turn. */
  private fun showExample() {
    val names = assets.list("examples")?.filter { it.endsWith(".png") }?.sorted().orEmpty()
    if (names.isEmpty()) return
    val name = names[exampleIndex % names.size]
    exampleIndex++
    vm.setPhoto(assets.open("examples/$name").use { it.readBytes() }, name)
  }

  private fun open(uri: Uri) {
    val name =
      contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c
        ->
        if (c.moveToFirst()) c.getString(0) else null
      } ?: "photo"
    val bytes =
      runCatching { contentResolver.openInputStream(uri)?.use { it.readBytes() } }.getOrNull()
    if (bytes == null) {
      Toast.makeText(this, "Cannot read that photo", Toast.LENGTH_SHORT).show()
      return
    }
    vm.setPhoto(bytes, name)
  }

  private fun copy(text: String) {
    getSystemService(ClipboardManager::class.java)
      .setPrimaryClip(ClipData.newPlainText("LaTeX", text))
    Log.i(MathViewModel.TAG, "COPIED chars=${text.length}")
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
      MathViewModel.TAG,
      "extras autoload=${i.getBooleanExtra("autoload", false)} " +
        "image=${i.getStringExtra("image")} mode=${i.getStringExtra("mode")} " +
        "plan=${i.getStringExtra("plan")}",
    )
    i.getStringExtra("mode")?.let { m -> runCatching { vm.setMode(Mode.valueOf(m)) } }
    if (i.getBooleanExtra("autoload", false) || i.hasExtra("plan")) vm.download()
    i.getStringExtra("image")?.let { path ->
      val f = File(root, path)
      if (f.isFile) vm.setPhoto(f.readBytes(), f.name)
      else Log.w(MathViewModel.TAG, "no image at ${f.path}")
    }
    i.getStringExtra("plan")?.let { AutoRun(vm, root).start(lifecycleScope, it) }
  }

  private companion object {
    val SCRIPT_EXTRAS = listOf("autoload", "image", "mode", "plan")
  }
}
