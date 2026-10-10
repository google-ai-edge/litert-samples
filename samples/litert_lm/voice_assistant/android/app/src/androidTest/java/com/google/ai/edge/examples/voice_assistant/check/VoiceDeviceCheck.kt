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
// samples/voice/src/androidTest/kotlin/io/github/johnrocky/hfmodels/check/VoiceDeviceCheck.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.check

import android.app.AlarmManager
import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.os.Build
import android.os.SystemClock
import android.provider.AlarmClock
import android.provider.Settings
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.examples.voice_assistant.VoiceAssistantException
import com.google.ai.edge.examples.voice_assistant.asr.LiteRtTranscriber
import com.google.ai.edge.examples.voice_assistant.asr.Transcriber
import com.google.ai.edge.examples.voice_assistant.data.DownloadStatus
import com.google.ai.edge.examples.voice_assistant.data.ModelCatalog
import com.google.ai.edge.examples.voice_assistant.data.ModelStore
import com.google.ai.edge.examples.voice_assistant.llm.ChatEngine
import com.google.ai.edge.examples.voice_assistant.loop.Endpointer
import com.google.ai.edge.examples.voice_assistant.loop.PhoneTools
import com.google.ai.edge.examples.voice_assistant.loop.SpeechPlayer
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop.Event
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoopConfig
import com.google.ai.edge.examples.voice_assistant.tts.LiteRtSpeaker
import com.google.ai.edge.examples.voice_assistant.tts.Speaker
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device check for the voice loop, in the app's own process. It proves on the connected phone what
 * the sample promises: the three models load from the app's store, one spoken command goes from
 * its audio to the phone's real tools (the alarm lands in the Clock app and Android reports it as
 * the next alarm) and out of the loudspeaker, and everything is released. It reports the network
 * state and does not switch it: run it in airplane mode to show that nothing needs the network.
 *
 * Before it runs: the models in the app's store (one load in the app, or the files side-loaded
 * into its external files dir, see the README), the calendar permissions, and the command's audio
 * (16 kHz mono 16-bit WAV):
 * ```
 * adb shell pm grant com.google.ai.edge.examples.voice_assistant android.permission.READ_CALENDAR
 * adb shell pm grant com.google.ai.edge.examples.voice_assistant android.permission.WRITE_CALENDAR
 * adb push c01.wav /data/local/tmp/voice-assistant/commands/c01.wav
 * ./gradlew :app:connectedDebugAndroidTest \
 *   -Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true \
 *   -Pandroid.testInstrumentationRunnerArguments.class=\
 * com.google.ai.edge.examples.voice_assistant.check.VoiceDeviceCheck
 * adb logcat -d -s voice-assistant-check | grep RESULT
 * ```
 * Keep the APKs installed: uninstalling the app deletes its model store. The last line is
 * `RESULT ok=true ...` when every step passed; the gradle task fails otherwise. Arguments: wav
 * (default /data/local/tmp/voice-assistant/commands/c01.wav, then <external files>/c01.wav),
 * expect (default "Set an alarm for seven thirty tomorrow morning."; compared on letters and
 * digits only).
 *
 * Android reports one next alarm, so no alarm may be set at or before the check's 07:30 (the
 * Quickstart's 06:45 included): the check stops first with `RESULT step=precondition ok=false`
 * and the alarm Android reports. The check sets a real alarm at 07:30 and then asks the Clock app
 * to dismiss it by its label; a Clock that does not honour that leaves it (the Samsung Clock did),
 * and the cleanup line names the label to turn off or delete by hand. Run it with the screen on and
 * unlocked, or with the app's screen over the keyguard: under the keyguard the app has no visible
 * activity and Android drops the Clock app's SET_ALARM activity start (BAL_BLOCK). The check brings
 * the app's launcher activity to the front with the extra `autoload=false`, which a debug build of
 * the sample takes as its scripted mode (it shows over the keyguard and loads nothing).
 */
@RunWith(AndroidJUnit4::class)
class VoiceDeviceCheck {
  private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
  private val args = InstrumentationRegistry.getArguments()
  private val wavArg = args.getString("wav")
  private val expect = args.getString("expect") ?: "Set an alarm for seven thirty tomorrow morning."

  @Test
  fun loadTurnAlarmRelease(): Unit = runBlocking {
    val failures = ArrayList<String>()
    fun step(name: String, ok: Boolean, values: String) {
      if (!ok) {
        failures += name
      }
      Log.i(TAG, "RESULT step=$name ok=$ok $values")
    }
    val net = network()
    val airplane =
      Settings.Global.getInt(ctx.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
    Log.i(
      TAG,
      "device=${Build.MODEL} build=${Build.DISPLAY} package=${ctx.packageName} network=$net " +
        "airplane_mode=$airplane",
    )
    // The app in front, as a user has it: the alarm intent is an activity start, and a visible app
    // may start one.
    ctx.packageManager.getLaunchIntentForPackage(ctx.packageName)?.let { launch ->
      val intent = launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK).putExtra("autoload", false)
      runCatching { InstrumentationRegistry.getInstrumentation().startActivitySync(intent) }
        .onFailure { Log.w(TAG, "could not bring the app to the front: $it") }
    }
    var asr: Transcriber? = null
    var tts: Speaker? = null
    var chat: ChatEngine? = null
    var player: SpeechPlayer? = null
    var label: String? = null
    try {
      // 0. the precondition: Android reports one next alarm, so an alarm at or before the check's
      // 07:30 would hide the one the check sets (or stand in for it).
      val alarms = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
      val checkAt = nextSevenThirty(System.currentTimeMillis())
      val already = alarms.nextAlarmClock?.triggerTime
      if (already != null && already < checkAt + 60_000) {
        throw PreconditionFailed(
          "Android's next alarm is ${hhmm(already)}, at or before the check's ${hhmm(checkAt)}; " +
            "turn it off or delete it in the Clock app and run the check again"
        )
      }

      // 1. load, from the app's store (the side-loaded copies are imported first)
      val catalog =
        ModelCatalog.parse(ctx.assets.open("models.json").bufferedReader().use { it.readText() })
      val store = ModelStore(File(ctx.filesDir, "models"))
      for (entry in catalog.models) {
        if (store.inspect(entry).status != DownloadStatus.READY) {
          store.sideLoad(entry, ctx.getExternalFilesDir(null))
        }
        if (store.inspect(entry).status != DownloadStatus.READY) {
          check(net != "none") { "${entry.id} is not in the store and there is no network" }
          store.download(entry) {}
        }
      }
      val zipformer = catalog.entry(ModelCatalog.ZIPFORMER)
      val kitten = catalog.entry(ModelCatalog.KITTEN)
      val gemma = catalog.entry(ModelCatalog.GEMMA)
      var t0 = SystemClock.elapsedRealtime()
      val a =
        LiteRtTranscriber.open(
            store.file(zipformer, "zipformer_ctc_fp16.tflite"),
            store.file(zipformer, "tokens.txt"),
          )
          .also { asr = it }
      val asrMs = SystemClock.elapsedRealtime() - t0
      t0 = SystemClock.elapsedRealtime()
      val symbols = { ctx.assets.open("symbols.json").bufferedReader().use { it.readText() } }
      val s = LiteRtSpeaker.open({ name -> store.file(kitten, name) }, symbols).also { tts = it }
      val ttsMs = SystemClock.elapsedRealtime() - t0
      t0 = SystemClock.elapsedRealtime()
      val c =
        ChatEngine.open(
            store.file(gemma, "gemma-4-E2B-it.litertlm"),
            File(ctx.cacheDir, "litertlm"),
          )
          .also { chat = it }
      val llmMs = SystemClock.elapsedRealtime() - t0
      step(
        "load",
        true,
        "asr=${zipformer.id}/${zipformer.backend} asr_ms=$asrMs " +
          "tts=${kitten.id}/${kitten.backend} tts_ms=$ttsMs " +
          "llm=${gemma.id}/${gemma.backend} llm_ms=$llmMs",
      )

      // 2. the command: its WAV through the loop's endpointer, the utterance to turn(pcm), the
      // phone's real tools
      val wav =
        listOfNotNull(
            wavArg?.let(::File),
            File("/data/local/tmp/voice-assistant/commands/c01.wav"),
            ctx.getExternalFilesDir(null)?.let { File(it, "c01.wav") },
          )
          .first { it.canRead() }
      val config = VoiceLoopConfig()
      val utterance = endpoint(readWav(wav), config.endpointer.hangoverMs)
      val p = SpeechPlayer(s.sampleRate).also { player = it }
      val loop = VoiceLoop(a, c, s, PhoneTools.all(ctx), config.copy(player = p))
      val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
      val before = am.nextAlarmClock?.triggerTime
      val events = withTimeout(TURN_TIMEOUT_MS) { loop.turn(utterance).toList() }
      val heard = (events.firstOrNull { it is Event.Heard } as Event.Heard?)?.text.orEmpty()
      val calls = events.filterIsInstance<Event.ToolCalled>()
      val alarm =
        calls.firstOrNull {
          it.name == "set_alarm" &&
            num(it.args["hour"]) == 7 &&
            num(it.args["minute"]) == 30 &&
            !it.result.startsWith("Error")
        }
      label = alarm?.args?.get("label")?.toString()
      val t = (events.lastOrNull() as? Event.Done)?.timing
      val speaking = events.count { it is Event.Speaking }
      val errors = events.filterIsInstance<Event.Error>()
      // The Clock app takes the intent on its own time: wait for Android to report the alarm.
      var after = am.nextAlarmClock?.triggerTime
      val until = SystemClock.elapsedRealtime() + ALARM_WAIT_MS
      while (alarm != null && !isSevenThirty(after) && SystemClock.elapsedRealtime() < until) {
        delay(200)
        after = am.nextAlarmClock?.triggerTime
      }
      val state = PhoneTools.phoneState(ctx)
      val nextLine = state.lineSequence().first()
      val alarmSet = alarm != null && isSevenThirty(after) && nextLine.endsWith("07:30")
      val heardOk = words(heard) == words(expect)
      val callText = calls.joinToString(",", "[", "]") { "${it.name}${it.args}" }
      val errorText =
        errors.joinToString(" | ", "[", "]") { "${it.code}: ${it.message.take(160)}" }
      step(
        "turn",
        heardOk && alarmSet && t?.firstAudioMs != null && speaking > 0 && errors.isEmpty(),
        "wav=${wav.path} heard=${q(heard)} expect=${q(expect)} heard_ok=$heardOk " +
          "calls=$callText tool_ok=${alarm != null} alarm_set=$alarmSet " +
          "next_alarm_before=${hhmm(before)} next_alarm_after=${hhmm(after)} " +
          "phone=${q(nextLine)} ms_first_audio=${f(t?.firstAudioMs)} " +
          "hangover_ms=${config.endpointer.hangoverMs} ms_total=${f(t?.totalMs)} " +
          "ms_transcribe=${f(t?.transcribeMs)} ms_first_token=${f(t?.firstTokenMs)} " +
          "speaking=$speaking errors=$errorText spoken=${q(t?.spoken.orEmpty())} " +
          "reply=${q(t?.reply.orEmpty())}",
      )

      // 3. the network, as reported (this check does not switch it)
      Log.i(TAG, "RESULT info network=$net airplane_mode=$airplane")

      // 4. release: the loop, the player, the three models
      val r0 = SystemClock.elapsedRealtime()
      loop.closeAndJoin()
      p.close()
      player = null
      c.closeAndJoin()
      chat = null
      s.closeAndJoin()
      tts = null
      a.closeAndJoin()
      asr = null
      val closeMs = SystemClock.elapsedRealtime() - r0
      step("release", closeMs < 10_000, "close_ms=$closeMs")
    } catch (e: PreconditionFailed) {
      step("precondition", false, "reason=${q(e.message.orEmpty())}")
    } catch (e: VoiceAssistantException) {
      step("exception", false, "error=${e.code} reason=${q(e.message.orEmpty())}")
    } catch (e: Exception) {
      Log.e(TAG, "failed", e)
      step("exception", false, "error=${e.javaClass.simpleName} message=${q(e.message.orEmpty())}")
    } finally {
      player?.close()
      runCatching { chat?.closeAndJoin() }
      runCatching { tts?.closeAndJoin() }
      runCatching { asr?.closeAndJoin() }
      dismiss(label)
      Log.i(
        TAG,
        "RESULT ok=${failures.isEmpty()} failed=$failures network=$net device=${Build.MODEL} " +
          "build=${Build.DISPLAY}",
      )
    }
    assertTrue("failed steps: $failures (adb logcat -d -s $TAG)", failures.isEmpty())
  }

  /**
   * Asks the Clock app to dismiss the check's alarm by its label and reports whether Android still
   * shows 07:30 next (an info line: no step passes or fails on it).
   */
  private suspend fun dismiss(label: String?) {
    if (label == null) {
      return
    }
    val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    val asked = runCatching {
      val intent =
        Intent(AlarmClock.ACTION_DISMISS_ALARM)
          .putExtra(AlarmClock.EXTRA_ALARM_SEARCH_MODE, AlarmClock.ALARM_SEARCH_MODE_LABEL)
          .putExtra(AlarmClock.EXTRA_MESSAGE, label)
          .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
      ctx.startActivity(intent)
    }
    val until = SystemClock.elapsedRealtime() + ALARM_WAIT_MS
    while (isSevenThirty(am.nextAlarmClock?.triggerTime) && SystemClock.elapsedRealtime() < until) {
      delay(200)
    }
    val left = isSevenThirty(am.nextAlarmClock?.triggerTime)
    val todo =
      if (left) {
        "left=true todo=" +
          q("turn off or delete the 07:30 alarm labelled '$label' in the Clock app by hand")
      } else {
        "left=false"
      }
    Log.i(
      TAG,
      "RESULT info cleanup dismiss_asked=${asked.isSuccess} " +
        "next_alarm=${hhmm(am.nextAlarmClock?.triggerTime)} $todo",
    )
  }

  /**
   * The WAV and the hangover's silence through an endpointer as the loop's listen uses it (20 ms
   * chunks): the utterance it cuts.
   */
  private fun endpoint(wav: FloatArray, hangoverMs: Int): FloatArray {
    val ep = VoiceLoopConfig().endpointer
    val all = wav + FloatArray(16000 * (hangoverMs + 200) / 1000)
    for (i in all.indices step 320) {
      val chunk = all.copyOfRange(i, minOf(i + 320, all.size))
      val cut =
        ep.feed(chunk).firstOrNull { it is Endpointer.Event.Utterance }
          as Endpointer.Event.Utterance?
      if (cut != null) {
        return cut.pcm
      }
    }
    return ep.flush()?.pcm ?: error("the endpointer found no speech in the WAV")
  }

  private fun network(): String {
    val cm = ctx.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    return if (cm.activeNetwork == null) "none" else "up"
  }

  private fun isSevenThirty(at: Long?): Boolean {
    if (at == null) {
      return false
    }
    val c = Calendar.getInstance().apply { timeInMillis = at }
    return c.get(Calendar.HOUR_OF_DAY) == 7 && c.get(Calendar.MINUTE) == 30
  }

  /** The next 07:30 after [now], local time. */
  private fun nextSevenThirty(now: Long): Long {
    val c =
      Calendar.getInstance().apply {
        timeInMillis = now
        set(Calendar.HOUR_OF_DAY, 7)
        set(Calendar.MINUTE, 30)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
      }
    if (c.timeInMillis <= now) {
      c.add(Calendar.DAY_OF_YEAR, 1)
    }
    return c.timeInMillis
  }

  private fun hhmm(at: Long?): String =
    at?.let { SimpleDateFormat("EEE-HH:mm", Locale.US).format(Date(it)) } ?: "none"

  private fun num(v: Any?): Int? =
    when (v) {
      is Number -> v.toInt()
      null -> null
      else -> v.toString().trim().toDoubleOrNull()?.toInt()
    }

  private fun words(s: String) = s.lowercase().filter { it.isLetterOrDigit() }

  private fun f(v: Double?) = v?.let { "%.0f".format(it) } ?: "none"

  private fun q(s: String) = "\"" + s.take(300).replace("\n", " ").replace("\"", "'") + "\""

  /** 16 kHz mono 16-bit PCM WAV -> floats in [-1, 1]; walks the RIFF chunks. */
  private fun readWav(f: File): FloatArray {
    val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    var p = 12
    while (p + 8 <= b.limit()) {
      val id = String(b.array(), p, 4, Charsets.US_ASCII)
      val n = b.getInt(p + 4)
      if (id == "data") {
        return FloatArray(n / 2) { i -> b.getShort(p + 8 + 2 * i) / 32768f }
      }
      p += 8 + n + (n and 1)
    }
    error("${f.name}: no data chunk")
  }

  private class PreconditionFailed(message: String) : Exception(message)

  private companion object {
    const val TAG = "voice-assistant-check"
    const val TURN_TIMEOUT_MS = 180_000L
    const val ALARM_WAIT_MS = 5_000L
  }
}
