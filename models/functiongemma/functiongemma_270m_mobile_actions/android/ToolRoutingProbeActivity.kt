package com.ai.edge.agent

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.tool
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ToolProvider
import java.io.File

/**
 * On-device tool-routing probe for the FunctionGemma 270M mobile-actions bundle.
 *
 * Mirrors `models/functiongemma/functiongemma_270m_mobile_actions/converted/verify_functiongemma_270m.py`
 * mode for mode: same tool schemas, same paired literal/paraphrase prompts, same
 * automaticToolCalling=false, greedy, one fresh conversation per prompt. Results are
 * logged as `FGP:` so the host can read them back with logcat.
 *
 * The model must be pushed to the app's files dir first:
 *   adb push function-gemma-q8-ekv1024.litertlm /data/local/tmp/
 *   adb shell run-as com.ai.edge.agent ... (or use the SAF-less path below)
 */
class ToolRoutingProbeActivity : Activity() {

  companion object {
    private const val TAG = "FGP"
    private const val MODEL = "function-gemma-q8-ekv1024.litertlm"
    private const val ROUNDS = 3

    /**
     * The four tools of the 4-tool cell that peaked on desktop, as the AAR's
     * `tool(OpenApiTool)` adapter wants them: a bare `name`/`description`/`parameters`
     * object. Verified by decompiling ToolKt$tool$2 — it reads `name` off the top
     * level of the parsed JSON, so the `{"type":"function","function":{...}}` envelope
     * used by the Python harness throws ToolException there. The description text and
     * the parameter schema are otherwise identical to the desktop run, which is the
     * part that matters for comparability.
     */
    private val TOOL_SCHEMAS = listOf(
      """{"name":"open_flashlight","description":"Turns the phone's flashlight on.","parameters":{"type":"object","properties":{},"required":[]}}""",
      """{"name":"close_flashlight","description":"Turns the phone's flashlight off.","parameters":{"type":"object","properties":{},"required":[]}}""",
      """{"name":"query_calendar","description":"Lists the events on the user's calendar for today.","parameters":{"type":"object","properties":{},"required":[]}}""",
      """{"name":"take_photo","description":"Takes a photo with the phone's camera.","parameters":{"type":"object","properties":{},"required":[]}}""",
    )

    /** prompt -> expected tool, expected kind. Same set as the desktop sensitivity mode. */
    private val PROBES = listOf(
      Triple("what is on my calendar today", "query_calendar", "literal"),
      Triple("whats on my calender today", "query_calendar", "paraphrase"),
      Triple("calendar today", "query_calendar", "terse"),
      Triple("what's my schedule", "query_calendar", "paraphrase"),
      Triple("list my calendar events", "query_calendar", "paraphrase"),
      Triple("do i have meetings today", "query_calendar", "paraphrase"),
      Triple("am i busy today", "query_calendar", "paraphrase"),
      Triple("show my events", "query_calendar", "paraphrase"),
      Triple("turn on the flashlight", "open_flashlight", "literal"),
      Triple("i need some light in here", "open_flashlight", "paraphrase"),
      Triple("it is dark in here", "open_flashlight", "paraphrase"),
      Triple("lights on please", "open_flashlight", "paraphrase"),
      Triple("turn off the flashlight", "close_flashlight", "literal"),
      Triple("kill the torch now", "close_flashlight", "paraphrase"),
      Triple("shut the light", "close_flashlight", "paraphrase"),
      Triple("no more light", "close_flashlight", "paraphrase"),
    )

    private val ARG_PROBES = listOf(
      Triple("take a photo", "take_photo", "literal"),
      Triple("snap a picture of the room", "take_photo", "paraphrase"),
      Triple("set an alarm for 07:30", "set_alarm", "literal"),
      Triple("wake me up at 06:00", "set_alarm", "paraphrase"),
      Triple("send a message to mom saying hi", "send_message", "paraphrase"),
      Triple("write a note titled groceries", "append_note", "literal"),
      Triple("what is the weather in nairobi", "noop", "paraphrase"),
      Triple("thanks, that was helpful", "noop", "literal"),
    )
  }

  /**
   * A tool defined by a raw OpenAPI schema string, mirroring SchemaTool in the Python
   * harness. `tool(OpenApiTool)` is the library's own adapter from a raw schema
   * to the ToolProvider that ConversationConfig wants, so there is no hand-rolled
   * provider and no reliance on the mangled `provideTools$...` member name.
   */
  private fun schemaTool(json: String): ToolProvider =
      tool(
          object : OpenApiTool {
            override fun getToolDescriptionJsonString(): String = json
            override fun execute(args: String): String = "ok"
          }
      )

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    Thread {
      try {
        run()
      } catch (t: Throwable) {
        Log.e(TAG, "PROBE_FAILED ${t.javaClass.name}: ${t.message}", t)
      }
    }.start()
  }

  private fun run() {
    val model = File(filesDir, MODEL)
    if (!model.exists()) {
      Log.e(TAG, "MODEL_MISSING ${model.absolutePath}")
      return
    }
    Log.i(TAG, "MODEL ${model.length()} bytes")
    Log.i(TAG, "DEVICE ${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT}")
    repeat(ROUNDS) { round -> runRound(model, round) }
    Log.i(TAG, "DONE")
  }

  /** VmHWM is the kernel's peak-RSS watermark, so it survives a GC mid-run. */
  private fun peakRssMb(): Long =
      File("/proc/self/status")
          .readLines()
          .first { it.startsWith("VmHWM:") }
          .split(Regex("\\s+"))[1]
          .toLong() / 1024

  private fun runRound(model: File, round: Int) {
    val results = mutableListOf<String>()
    Log.i(TAG, "ROUND_BEGIN $round rss_before_mb=${peakRssMb()}")

    val engine = Engine(
        EngineConfig(
            modelPath = model.absolutePath,
            backend = Backend.CPU(),
            maxNumTokens = 2048,
            cacheDir = File(cacheDir, "litert").absolutePath,
        )
    )
    val t0 = System.nanoTime()
    engine.initialize()
    val initMs = (System.nanoTime() - t0) / 1_000_000
    Log.i(TAG, "ROUND $round INIT_MS $initMs rss_after_init_mb=${peakRssMb()}")

    val tools = TOOL_SCHEMAS.map { schemaTool(it) }
    var hits = 0
    var scored = 0

    for ((prompt, expected, kind) in PROBES + ARG_PROBES) {
      val inSet = TOOL_SCHEMAS.any { it.contains("\"${expected}\"") }
      if (!inSet) continue // the 4-tool cell cannot score these; keep the table honest
      scored++
      val start = System.nanoTime()
      var got: String? = null
      var prose = ""
      try {
        engine.createConversation(
            ConversationConfig(
                tools = tools,
                automaticToolCalling = false,
                samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0, seed = 0),
            )
        ).use { convo ->
          val reply = convo.sendMessage(prompt)
          val call = reply.toolCalls?.firstOrNull()
          if (call != null) {
            got = call.name
          } else {
            prose = reply.contents?.contents?.filterIsInstance<Content.Text>()?.joinToString("") { it.text }?.trim().orEmpty()
          }
        }
      } catch (t: Throwable) {
        prose = "EXCEPTION ${t.javaClass.simpleName}: ${t.message}"
      }
      val ms = (System.nanoTime() - start) / 1_000_000
      val ok = got == expected
      if (ok) hits++
      val line = "$round|$kind|$expected|$got|$ok|$ms|${prose.replace("|", "/").take(80)}|$prompt"
      results.add(line)
      Log.i(TAG, "ROW $line")
    }

    Log.i(
        TAG,
        "ROUND_SUMMARY round=$round scored=$scored hits=$hits " +
            "acc=$hits/$scored peak_rss_mb=${peakRssMb()}"
    )
    engine.close()
    Log.i(TAG, "ROUND_END $round")
  }
}