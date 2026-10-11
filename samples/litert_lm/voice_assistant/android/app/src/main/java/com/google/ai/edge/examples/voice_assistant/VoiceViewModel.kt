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
// samples/voice/src/main/kotlin/io/github/johnrocky/hfmodels/samples/voice/VoiceViewModel.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant

import android.app.Application
import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.examples.voice_assistant.asr.LiteRtTranscriber
import com.google.ai.edge.examples.voice_assistant.asr.Transcriber
import com.google.ai.edge.examples.voice_assistant.data.DownloadSafety
import com.google.ai.edge.examples.voice_assistant.data.DownloadState
import com.google.ai.edge.examples.voice_assistant.data.DownloadStatus
import com.google.ai.edge.examples.voice_assistant.data.ModelCatalog
import com.google.ai.edge.examples.voice_assistant.data.ModelEntry
import com.google.ai.edge.examples.voice_assistant.data.ModelStore
import com.google.ai.edge.examples.voice_assistant.llm.ChatEngine
import com.google.ai.edge.examples.voice_assistant.loop.Endpointer
import com.google.ai.edge.examples.voice_assistant.loop.MicSource
import com.google.ai.edge.examples.voice_assistant.loop.PhoneTools
import com.google.ai.edge.examples.voice_assistant.loop.SpeechPlayer
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop.Event
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoopConfig
import com.google.ai.edge.examples.voice_assistant.tts.LiteRtSpeaker
import com.google.ai.edge.examples.voice_assistant.tts.Speaker
import com.google.ai.edge.examples.voice_assistant.ui.ModelRow
import com.google.ai.edge.examples.voice_assistant.ui.VoiceUi
import com.google.ai.edge.examples.voice_assistant.ui.ms
import com.google.ai.edge.examples.voice_assistant.ui.on
import com.google.ai.edge.examples.voice_assistant.ui.size
import java.io.File
import java.util.Locale
import kotlin.math.sqrt
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The three models (downloaded into the app's private storage on first use), one [VoiceLoop] over
 * them with the phone's real tools, and the microphone. The loop runs off the main thread; [ui] is
 * what the screen draws.
 */
class VoiceViewModel(private val app: Application) : AndroidViewModel(app) {
  private val catalog =
    ModelCatalog.parse(app.assets.open("models.json").bufferedReader().use { it.readText() })
  private val store = ModelStore(File(app.filesDir, "models"))
  // Written by the load on a background thread, read by the screen's calls on the main thread.
  @Volatile private var asr: Transcriber? = null
  @Volatile private var tts: Speaker? = null
  @Volatile private var chat: ChatEngine? = null
  @Volatile private var player: SpeechPlayer? = null
  @Volatile private var loop: VoiceLoop? = null
  private val cleanupScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
  private var loadJob: Job? = null
  private var listenJob: Job? = null
  private var turnJob: Job? = null
  private val afterLoad = ArrayList<() -> Unit>()

  /**
   * Whether the screen is visible (the activity sets it): what was to start once the models are
   * loaded starts only then, so a load that ends with the app in the background stays silent.
   */
  @Volatile var visible = true
  private var tap: MicTap? = null
  private var recorder: TurnRecorder? = null
  private var config = VoiceLoopConfig()

  private val _ui =
    MutableStateFlow(
      VoiceUi(models = catalog.models.map { ModelRow(it.id, it.role, it.model, it.totalBytes) })
    )
  val ui: StateFlow<VoiceUi> = _ui

  init {
    viewModelScope.launch {
      for (entry in catalog.models) {
        setState(entry, store.inspect(entry))
      }
      val net = network()
      val airplane = airplaneMode()
      _ui.update { it.copy(network = net, airplaneMode = airplane) }
    }
  }

  /**
   * Makes the three models ready (the side-loaded copies, else a download of the missing files),
   * then loads the transcriber, the speaker and the chat engine in turn (once); then runs [then]
   * (on the main thread, as the caller). On a metered network a download waits for the user's
   * confirmation when [askOnMetered] (the screen's button); the script extras, a debug-build tool,
   * do not ask.
   */
  fun load(askOnMetered: Boolean = true, then: (() -> Unit)? = null) {
    if (loop != null) {
      then?.invoke()
      return
    }
    then?.let { afterLoad += it }
    if (loadJob?.isActive == true) {
      return
    }
    _ui.update { it.copy(loading = true, status = "Loading…", error = null) }
    loadJob =
      viewModelScope.launch(Dispatchers.Default) {
        val t0 = SystemClock.elapsedRealtime()
        try {
          val missing = catalog.models.filter { !fromStore(it) }
          if (missing.isNotEmpty()) {
            if (network() == "none") {
              throw VoiceAssistantException(
                "DOWNLOAD_FAILED",
                "${missing.first().model} is not on the phone and there is no network to " +
                  "download it",
              )
            }
            if (askOnMetered && isMetered()) {
              val bytes = missing.sumOf { it.totalBytes - store.inspect(it).receivedBytes }
              _ui.update {
                it.copy(
                  loading = false,
                  status = "Waiting for the download to be confirmed",
                  downloadConfirmation = bytes,
                )
              }
              return@launch
            }
            for (entry in missing) {
              download(entry)
            }
          }
          val zipformer = catalog.entry(ModelCatalog.ZIPFORMER)
          val kitten = catalog.entry(ModelCatalog.KITTEN)
          val gemma = catalog.entry(ModelCatalog.GEMMA)
          _ui.update { it.copy(status = "Transcriber: starting on gpu") }
          val a =
            LiteRtTranscriber.open(
                store.file(zipformer, "zipformer_ctc_fp16.tflite"),
                store.file(zipformer, "tokens.txt"),
              )
              .also { asr = it }
          _ui.update { it.copy(status = "Speaker: starting on cpu") }
          val symbols = { app.assets.open("symbols.json").bufferedReader().use { it.readText() } }
          val s =
            LiteRtSpeaker.open({ name -> store.file(kitten, name) }, symbols).also { tts = it }
          _ui.update { it.copy(status = "Chat model: starting on gpu") }
          val c =
            ChatEngine.open(
                store.file(gemma, "gemma-4-E2B-it.litertlm"),
                File(app.cacheDir, "litertlm"),
              )
              .also { chat = it }
          val p = SpeechPlayer(s.sampleRate).also { player = it }
          loop = VoiceLoop(a, c, s, PhoneTools.all(app), config.copy(player = p))
          val loadMs = SystemClock.elapsedRealtime() - t0
          Log.i(
            TAG,
            "ready asr=${zipformer.id}/${zipformer.backend} tts=${kitten.id}/${kitten.backend} " +
              "llm=${gemma.id}/${gemma.backend} load_ms=$loadMs network=${network()}",
          )
          val status = "Ready · loaded in ${ms(loadMs.toDouble())}"
          _ui.update { it.copy(loading = false, ready = true, status = status) }
          refreshPhoneState()
          // afterLoad is touched on the main thread only.
          withContext(Dispatchers.Main) {
            val pending = ArrayList(afterLoad).also { afterLoad.clear() }
            if (visible) {
              pending.forEach { it() }
            }
          }
        } catch (e: CancellationException) {
          releaseLoaded()
          throw e
        } catch (e: VoiceAssistantException) {
          Log.e(TAG, "load failed ${e.code}: ${e.message}", e)
          releaseLoaded()
          _ui.update { it.copy(loading = false, ready = false, status = "${e.code}: ${e.message}") }
        } catch (e: Exception) {
          Log.e(TAG, "load failed", e)
          releaseLoaded()
          val status = "${e.javaClass.simpleName}: ${e.message}"
          _ui.update { it.copy(loading = false, ready = false, status = status) }
        }
      }
  }

  /**
   * Closes what a failed or cancelled load opened, the last first, and forgets it, so the next
   * load starts clean instead of opening the models a second time.
   */
  private suspend fun releaseLoaded() {
    val p = player
    val c = chat
    val s = tts
    val a = asr
    loop = null
    player = null
    chat = null
    tts = null
    asr = null
    withContext(NonCancellable) {
      runCatching { p?.close() }.onFailure { Log.w(TAG, "player close: ${it.message}") }
      runCatching { c?.closeAndJoin() }.onFailure { Log.w(TAG, "chat close: ${it.message}") }
      runCatching { s?.closeAndJoin() }.onFailure { Log.w(TAG, "speaker close: ${it.message}") }
      runCatching { a?.closeAndJoin() }.onFailure { Log.w(TAG, "asr close: ${it.message}") }
    }
  }

  /**
   * Whether the entry's files are in the store: already there, or imported from side-loaded
   * copies (development only), with a failed import on the status line. Without the free space for
   * the missing bytes, NOT_ENOUGH_STORAGE (a partial file stays for the next try).
   */
  private suspend fun fromStore(entry: ModelEntry): Boolean {
    var state = store.inspect(entry)
    if (state.status != DownloadStatus.READY) {
      ensureSpace(entry, state)
      val failure =
        try {
          store.sideLoad(entry, app.getExternalFilesDir(null)).skipped.firstOrNull()
        } catch (e: CancellationException) {
          throw e
        } catch (e: Exception) {
          "${e.javaClass.simpleName}: ${e.message}"
        }
      if (failure != null) {
        Log.w(TAG, "side-load of ${entry.id} failed: $failure")
        _ui.update {
          it.copy(
            status = "${entry.role}: side-load failed: $failure; downloading",
            error = "${entry.model}: side-load failed: $failure",
          )
        }
      }
      state = store.inspect(entry)
    }
    setState(entry, state)
    return state.status == DownloadStatus.READY
  }

  /** Downloads the entry's missing files; a failure is DOWNLOAD_FAILED. */
  private suspend fun download(entry: ModelEntry) {
    val state = store.inspect(entry)
    ensureSpace(entry, state)
    _ui.update { it.copy(status = "${entry.role}: downloading") }
    try {
      store.download(entry) { setState(entry, it) }
    } catch (e: CancellationException) {
      throw e
    } catch (e: Exception) {
      setState(entry, state.copy(status = DownloadStatus.ERROR, error = e.message))
      throw VoiceAssistantException(
        "DOWNLOAD_FAILED",
        "${entry.model}: ${e.javaClass.simpleName}: ${e.message}",
        e,
      )
    }
    setState(entry, store.inspect(entry))
  }

  /** The user's answer to the metered-network question: download now. */
  fun confirmDownload() {
    _ui.update { it.copy(downloadConfirmation = null) }
    load(askOnMetered = false)
  }

  /** The user's answer to the metered-network question: not now. */
  fun cancelDownload() {
    afterLoad.clear()
    _ui.update {
      it.copy(downloadConfirmation = null, status = "Not loaded: the download was cancelled")
    }
  }

  private fun isMetered(): Boolean {
    val cm = app.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    return cm.isActiveNetworkMetered
  }

  /** The model zoo app's rule: twice the bytes still missing must be free in filesDir. */
  private fun ensureSpace(entry: ModelEntry, state: DownloadState) {
    val missing = entry.totalBytes - state.receivedBytes
    if (missing <= 0) {
      return
    }
    val available = app.filesDir.usableSpace
    if (!DownloadSafety.hasSpace(missing, available)) {
      val needed = size(DownloadSafety.requiredFreeBytes(missing))
      throw VoiceAssistantException(
        "NOT_ENOUGH_STORAGE",
        "${entry.model} needs $needed free (twice the ${size(missing)} still missing); " +
          "${size(available)} is free",
      )
    }
  }

  private fun setState(entry: ModelEntry, state: DownloadState) {
    _ui.update { ui ->
      ui.copy(models = ui.models.map { if (it.id == entry.id) it.copy(state = state) else it })
    }
  }

  /** Opens the microphone and takes a turn per utterance until [stop]; a second press stops it. */
  fun toggleListen() {
    if (stillRunning(listenJob)) {
      stop()
    } else {
      listen()
    }
  }

  fun listen() {
    val l = loop
    if (l == null) {
      load(askOnMetered = false) { listen() }
      return
    }
    if (stillRunning(listenJob)) {
      return
    }
    val mic = MicSource()
    // The record mode writes each turn's utterance: only then is the last audio kept.
    val t = if (recorder != null) MicTap(mic.sampleRate) else null
    tap = t
    _ui.update { it.copy(listening = true, status = "Listening…", error = null) }
    // While recording: the loudest 20 ms of each second against the endpointer's start level, to
    // tell a quiet room from a microphone that hears nothing.
    var peak = 0.0
    var since = SystemClock.elapsedRealtime()
    listenJob =
      viewModelScope.launch(Dispatchers.Default) {
        try {
          val audio =
            mic.chunks().onEach { c ->
              if (t != null) {
                t.add(c)
                peak = maxOf(peak, sqrt(c.sumOf { (it * it).toDouble() } / c.size))
                val now = SystemClock.elapsedRealtime()
                if (now - since >= 1000) {
                  val level = String.format(Locale.US, "%.4f", peak)
                  Log.i(TAG, "mic_level max_rms=$level start_rms=${config.endpointer.startRms}")
                  peak = 0.0
                  since = now
                }
              }
            }
          l.listen(audio).collect { e -> onEvent(e, fromMic = true) }
        } catch (e: CancellationException) {
          throw e
        } catch (e: Exception) {
          Log.e(TAG, "listen failed", e)
          _ui.update { it.copy(error = "${e.javaClass.simpleName}: ${e.message}") }
        } finally {
          _ui.update {
            it.copy(listening = false, busy = false, status = if (it.ready) "Ready" else it.status)
          }
        }
      }
  }

  /** One turn from typed text (no microphone); a typed turn still running stops first. */
  fun say(text: String) {
    val l = loop
    if (l == null) {
      load(askOnMetered = false) { say(text) }
      return
    }
    val previous = turnJob
    previous?.cancel()
    turnJob =
      viewModelScope.launch(Dispatchers.Default) {
        // The loop already runs one turn at a time; this wait keeps the earlier turn's last writes
        // (the screen state its TURN line reads, its record) apart from this turn's events.
        previous?.join()
        try {
          l.turn(text).collect { e -> onEvent(e, fromMic = false) }
        } catch (e: CancellationException) {
          throw e
        } catch (e: Exception) {
          Log.e(TAG, "turn failed", e)
          _ui.update { it.copy(busy = false, error = "${e.javaClass.simpleName}: ${e.message}") }
        }
      }
  }

  /** Stops the microphone and any turn in progress (the model and the sound stop with it). */
  fun stop() {
    listenJob?.cancel()
    turnJob?.cancel()
  }

  /**
   * The endpointer's start level ([Endpointer.startRms]; 0.02 is a voice toward the phone, sound
   * through a speaker needs less). A loaded loop is built again with it; a listen in progress
   * stops.
   */
  fun startRms(v: Float) {
    val e = config.endpointer
    val endpointer =
      Endpointer(e.sampleRate, v, e.startMs, e.hangoverMs, e.maxUtteranceMs, e.frameMs, e.preRollMs)
    config = config.copy(endpointer = endpointer)
    Log.i(TAG, "start_rms=$v")
    val old = loop ?: return
    stop()
    old.close()
    val a = asr ?: return
    val c = chat ?: return
    val s = tts ?: return
    loop = VoiceLoop(a, c, s, PhoneTools.all(app), config.copy(player = player))
  }

  /** Each turn's sound and events go under `<external files>/record/<name>/<turn>/`; null stops. */
  fun record(name: String?) {
    recorder =
      name?.let { n ->
        val root = File(app.getExternalFilesDir(null), "record/${File(n).name}").apply { mkdirs() }
        Log.i(TAG, "record to ${root.path}")
        TurnRecorder(root)
      }
  }

  private suspend fun onEvent(e: Event, fromMic: Boolean) {
    val hangover = if (fromMic) config.endpointer.hangoverMs else 0
    _ui.update { it.on(e, hangover) }
    recorder?.event(e, fromMic)
    Log.i(TAG, describe(e))
    // The turn is over: its TURN line and record are written whole even if the screen goes away
    // now (stop() cancels the job this runs in).
    if (e is Event.Done) {
      withContext(NonCancellable) { afterTurn(e.timing, fromMic, hangover) }
    }
  }

  private suspend fun afterTurn(t: VoiceLoop.TurnTiming, fromMic: Boolean, hangover: Int) {
    val state = refreshPhoneState()
    val calls = _ui.value.tools.joinToString(",", "[", "]") { "${it.call}->\"${it.result}\"" }
    val phone = state.lineSequence().firstOrNull() ?: ""
    Log.i(
      TAG,
      "TURN input=${if (fromMic) "mic" else "text"} heard=${q(t.heard)} tools=$calls " +
        "ms_first_audio=${f(t.firstAudioMs)} hangover_ms=$hangover " +
        "ms_end_of_speech_to_sound=${f(t.firstAudioMs?.let { it + hangover })} " +
        "ms_transcribe=${f(t.transcribeMs)} ms_first_token=${f(t.firstTokenMs)} " +
        "ms_first_sentence=${f(t.firstSentenceMs)} ms_reply=${f(t.replyMs)} " +
        "ms_total=${f(t.totalMs)} spoken=${q(t.spoken)} reply=${q(t.reply)} phone=${q(phone)} " +
        "network=${network()}",
    )
    val r = recorder ?: return
    val s = tts ?: return
    // reply.wav: the sentences said, synthesized again (the loop plays them and keeps no copy).
    val audio = r.sentences().map { s.synthesize(it, config.voice, config.speed).samples }
    val firstWrite = player?.firstWriteAtNanos ?: 0L
    val dir = r.finish(tap.takeIf { fromMic }, audio, s.sampleRate, firstWrite, state)
    Log.i(TAG, "recorded ${dir?.path}")
  }

  private suspend fun refreshPhoneState(): String {
    val s =
      try {
        PhoneTools.phoneState(app)
      } catch (e: CancellationException) {
        throw e
      } catch (e: Exception) {
        "?"
      }
    val net = network()
    val airplane = airplaneMode()
    _ui.update { it.copy(phoneState = s, network = net, airplaneMode = airplane) }
    return s
  }

  /** What the phone's network is: none in airplane mode. */
  fun network(): String {
    val cm = app.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    val n = cm.activeNetwork ?: return "none"
    val caps = cm.getNetworkCapabilities(n) ?: return "unknown"
    val kinds =
      listOf(
        NetworkCapabilities.TRANSPORT_CELLULAR to "cellular",
        NetworkCapabilities.TRANSPORT_WIFI to "wifi",
        NetworkCapabilities.TRANSPORT_ETHERNET to "ethernet",
        NetworkCapabilities.TRANSPORT_VPN to "vpn",
      )
    return kinds
      .filter { caps.hasTransport(it.first) }
      .joinToString("+") { it.second }
      .ifEmpty { "other" }
  }

  private fun airplaneMode(): Boolean =
    Settings.Global.getInt(app.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1

  override fun onCleared() {
    val l = loop
    loop = null
    // The models outlive viewModelScope: they are closed on a scope of their own, as the model zoo
    // app does, and the scope ends with them.
    cleanupScope.launch {
      try {
        l?.closeAndJoin()
        releaseLoaded()
      } catch (failure: Throwable) {
        if (failure is CancellationException) {
          throw failure
        }
        Log.e(TAG, "Model cleanup failed", failure)
      } finally {
        cleanupScope.cancel()
      }
    }
    super.onCleared()
  }

  companion object {
    const val TAG = "VoiceAssistant"

    private fun describe(e: Event): String =
      when (e) {
        is Event.Heard -> {
          "event=Heard text=${q(e.text)} audio_ms=${f(e.audioMs)} " +
            "transcribe_ms=${f(e.transcribeMs)}"
        }
        is Event.ToolCalled -> {
          "event=ToolCalled name=${e.name} args=${e.args} result=${q(e.result)} ms=${f(e.ms)}"
        }
        is Event.Speaking -> {
          "event=Speaking sentence=${q(e.sentence)} synth_ms=${f(e.synthMs)} " +
            "first_audio_ms=${f(e.firstAudioMs)}"
        }
        is Event.Error -> "event=Error code=${e.code} message=${q(e.message)}"
        is Event.Done -> "event=Done total_ms=${f(e.timing.totalMs)}"
        else -> "event=$e"
      }

    private fun f(v: Double?) = v?.let { String.format(Locale.US, "%.0f", it) } ?: "none"

    private fun q(s: String) = "\"" + s.replace("\n", "\\n").replace("\"", "'") + "\""
  }
}

/**
 * Whether a listen still holds the microphone and the loop: until its job has completed, not only
 * while it is active. A stopped listen is no longer active at once but unwinds for a while (the
 * runtime confirms the model's stop, the conversation closes, the sentence in synthesis finishes);
 * a listen started meanwhile opens a second microphone, and the first one's end then shows the
 * screen as idle while the second one listens.
 */
internal fun stillRunning(job: Job?): Boolean = job?.isCompleted == false
