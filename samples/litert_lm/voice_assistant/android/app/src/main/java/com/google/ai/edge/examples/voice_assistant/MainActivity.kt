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

// Adapted from john-rocky/hfmodels-android (commit 3086d647):
// samples/voice/src/main/kotlin/io/github/johnrocky/hfmodels/samples/voice/MainActivity.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant

import android.Manifest
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.google.ai.edge.examples.voice_assistant.loop.Endpointer
import com.google.ai.edge.examples.voice_assistant.ui.VoiceScreen

/**
 * The voice loop's screen. Launch extras, for a scripted take in a debuggable build (a release
 * build ignores them; all optional; a running screen takes them again):
 * - `--ez autoload true`: load the three models now (downloading what is missing);
 * - `--ez autolisten true`: open the microphone once they are loaded;
 * - `--es say "<text>"`: one turn from this text once they are loaded (no microphone);
 * - `--es record <name>`: write each turn's sound and events under
 *   `<external files>/record/<name>/<turn>/`;
 * - `--ef start_rms 0.01`: the endpointer's start level (default
 *   [Endpointer.DEFAULT_START_RMS], a voice toward the phone; less for a speaker).
 *
 * Scripted mode (any of these extras present, whatever its value, in a debuggable build) shows
 * the screen over the keyguard, turns the display on and keeps it on; a normal launch keeps it on
 * only while the models load, the microphone is open or a request runs. Under the keyguard the
 * activity is not visible: Android drops the Clock app's SET_ALARM activity start from the app
 * (BAL_BLOCK, result code 102; Galaxy S26, 2026-10-03) and the hidden activity's process runs in
 * the background cpuset. While the screen is not visible, the microphone and any request in
 * progress stop.
 */
class MainActivity : ComponentActivity() {
  private val vm: VoiceViewModel by viewModels()

  /** Set by a scripted launch for the rest of the activity's life, as the keyguard flags are. */
  private var scriptedMode by mutableStateOf(false)

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    if (scripted(intent)) {
      overKeyguard()
    }
    enableEdgeToEdge()
    val missing =
      PERMISSIONS.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
    if (missing.isNotEmpty()) {
      requestPermissions(missing.toTypedArray(), 1)
    }
    setContent {
      val ui by vm.ui.collectAsState()
      // The screen stays on while the models load, the microphone is open or a request runs: a
      // locked phone moves a hidden activity's process to the background cpuset (little cores).
      // Scripted mode keeps it on throughout: VoiceDeviceCheck brings this screen up idle
      // (autoload=false) while it runs its own models, and its alarm needs the app visible.
      val keepOn = scriptedMode || ui.keepsScreenOn
      LaunchedEffect(keepOn) {
        if (keepOn) {
          window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
          window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
      }
      VoiceScreen(
        ui,
        onMic = { vm.toggleListen() },
        onLoad = { vm.load() },
        onConfirmDownload = { vm.confirmDownload() },
        onCancelDownload = { vm.cancelDownload() },
      )
    }
    if (savedInstanceState == null) {
      handle(intent)
    }
  }

  override fun onStart() {
    super.onStart()
    vm.visible = true
  }

  override fun onStop() {
    super.onStop()
    vm.visible = false
    vm.stop()
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    if (scripted(intent)) {
      overKeyguard()
    }
    handle(intent)
  }

  /** The script extras are for development: a release build ignores them. */
  private fun debuggable(): Boolean =
    (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0

  private fun scripted(i: Intent) = debuggable() && SCRIPT_EXTRAS.any { i.hasExtra(it) }

  private fun overKeyguard() {
    setShowWhenLocked(true)
    setTurnScreenOn(true)
    scriptedMode = true
  }

  private fun handle(i: Intent) {
    if (!debuggable()) {
      return
    }
    val say = i.getStringExtra("say")
    val listen = i.getBooleanExtra("autolisten", false)
    i.getStringExtra("record")?.let { vm.record(it) }
    if (i.hasExtra("start_rms")) {
      vm.startRms(i.getFloatExtra("start_rms", Endpointer.DEFAULT_START_RMS))
    }
    val autoload = i.getBooleanExtra("autoload", false)
    Log.i(
      VoiceViewModel.TAG,
      "extras autoload=$autoload autolisten=$listen say=${say != null} " +
        "record=${i.getStringExtra("record")}",
    )
    // The script extras are a debug-build tool: a download they start does not ask first, even on
    // a metered network.
    when {
      say != null -> vm.load(askOnMetered = false) { vm.say(say) }
      listen -> vm.load(askOnMetered = false) { vm.listen() }
      autoload -> vm.load(askOnMetered = false)
    }
  }

  private companion object {
    val PERMISSIONS =
      listOf(
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.READ_CALENDAR,
        Manifest.permission.WRITE_CALENDAR,
      )
    val SCRIPT_EXTRAS = listOf("autoload", "autolisten", "say", "record", "start_rms")
  }
}
