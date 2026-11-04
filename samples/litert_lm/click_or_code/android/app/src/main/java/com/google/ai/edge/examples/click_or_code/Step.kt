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

import org.json.JSONArray
import org.json.JSONObject

/** One `<tool_call>` of a reply, as the screen shows it. */
data class ToolCall(
  /** `computer_use` or `bash`, as the model wrote it. */
  val name: String,
  val action: String? = null,
  /** The point in the model's 0-999 coordinates, for a click-like action. */
  val coordinate: Pair<Double, Double>? = null,
  val start: Pair<Double, Double>? = null,
  val end: Pair<Double, Double>? = null,
  val text: String? = null,
  val keys: List<String> = emptyList(),
  val command: String? = null,
  /** False when the call's JSON did not parse and the call was read by the fallback. */
  val strictJson: Boolean = true,
) {
  val isCode: Boolean
    get() = name == "bash"
}

/** What a reply says: the reasoning, the one-sentence action, the tool calls. */
data class StepReply(
  val thinking: String,
  val actionLine: String?,
  val calls: List<ToolCall>,
) {
  val first: ToolCall?
    get() = calls.firstOrNull()
}

/**
 * The agent's step, as the model's own hybrid agent sends it (microsoft/Mage-VL-Cowork,
 * osworld_eval/mm_agents/mage_hybrid_agent.py), and the reading of its reply.
 */
object Step {
  /** The tokens the file holds: the system prompt, the screenshot, the request and the reply. */
  const val FILE_TOKENS = 4096

  /** What the runtime says when a prompt does not fit the file. */
  const val TOO_LONG_ERROR = "Prefill input length exceeds"

  /**
   * The reply's cap: what is left of [FILE_TOKENS] after the 1,095 tokens of the system prompt, the
   * 2,040 of the screenshot and at least 170 of the request, the screenshot step and the template's
   * framing.
   */
  const val MAX_OUTPUT_TOKENS = FILE_TOKENS - 1095 - 2040 - 170

  /** SHA-256 of the system prompt in `assets/`, the verbatim MAGE_HYBRID_SYSTEM_PROMPT (4,230 bytes). */
  const val SYSTEM_SHA256 = "db6aefb8923bfae10d6548d1f968b0b08be02825a6b171643bec8731351ecb36"

  /**
   * The step that comes before the one the app asks for: the model asks for a screenshot. On a first
   * step the model does that whatever the request, so the app starts at the second step of the
   * agent's conversation, with the user's screenshot as the answer to this call. The sentence after
   * `Action:` and the call are the model's own words on a first step.
   */
  const val SCREENSHOT_STEP =
    "I’m at the first step, and I have not seen the screen yet. A screenshot will show me the " +
      "current state before I do anything else.\n</think>\n\nAction: Take a screenshot of the " +
      "current screen.\n\n<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": " +
      "\"screenshot\"}}\n</tool_call>"

  /** The user turn before [SCREENSHOT_STEP]: the agent's step text with its screenshot collapsed. */
  fun requestTurn(request: String): String =
    "[This screenshot has been collapsed.]\nPlease generate the next move according to the UI " +
      "screenshot, instruction and previous actions.\n\nInstruction: ${request.trim()}\n\n" +
      "Previous actions:\nNone"

  /** Around the screenshot in the user turn after [SCREENSHOT_STEP]. */
  const val RESULT_BEFORE = "<tool_response>\n"
  const val RESULT_AFTER = "\n</tool_response>"

  private val TOOL_CALL = Regex("<tool_call>\\s*(.*?)\\s*(?:</tool_call>|$)", RegexOption.DOT_MATCHES_ALL)
  private val ACTION = Regex("(?m)^\\s*Action:\\s*(.+)$")
  private val COORD = Regex("\"coordinate\"\\s*:\\s*\\[\\s*(-?[\\d.]+)\\s*,\\s*(-?[\\d.]+)\\s*]")
  private val ACTION_ARG = Regex("\"action\"\\s*:\\s*\"(\\w+)\"")
  private val NAME = Regex("\"name\"\\s*:\\s*\"(\\w+)\"")
  private val COMMAND = Regex("\"command\"\\s*:\\s*\"(.*)\"\\s*}\\s*}?\\s*$", RegexOption.DOT_MATCHES_ALL)

  /** The model can write a typographic quote where JSON needs a straight one. */
  private fun straighten(s: String) = s.replace('“', '"').replace('”', '"')

  /** Reads [text], a whole reply or the part streamed so far. */
  fun parse(text: String): StepReply {
    val end = text.indexOf("</think>")
    val thinking = (if (end >= 0) text.substring(0, end) else text).removePrefix("<think>").trim()
    val after = if (end >= 0) text.substring(end + "</think>".length) else ""
    val action = ACTION.find(after)?.groupValues?.get(1)?.trim()
    val calls = TOOL_CALL.findAll(after).mapNotNull { readCall(it.groupValues[1]) }.toList()
    return StepReply(thinking, action, calls)
  }

  /** One call's JSON; when it does not parse, the fields the app shows, read one by one. */
  fun readCall(body: String): ToolCall? {
    val json = straighten(body)
    val obj = runCatching { JSONObject(json) }.getOrNull()
    if (obj != null) {
      val name = obj.optString("name").ifEmpty { null } ?: return null
      val args = obj.optJSONObject("arguments") ?: JSONObject()
      return ToolCall(
        name = name,
        action = args.optString("action").ifEmpty { null },
        coordinate = pair(args.optJSONArray("coordinate")),
        start = pair(args.optJSONArray("start")),
        end = pair(args.optJSONArray("end")),
        text = args.optString("text").ifEmpty { null },
        keys = strings(args.optJSONArray("keys")),
        command = args.optString("command").ifEmpty { null },
      )
    }
    val name = NAME.find(json)?.groupValues?.get(1)
    val command = COMMAND.find(json)?.groupValues?.get(1)
    if (command != null && name != null) {
      val plain = command.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\")
      return ToolCall(name = name, command = plain, strictJson = false)
    }
    val c = COORD.find(json)
    val a = ACTION_ARG.find(json)
    if (c == null && a == null) return null
    return ToolCall(
      name = name ?: "computer_use",
      action = a?.groupValues?.get(1),
      coordinate =
        c?.let {
          val x = it.groupValues[1].toDoubleOrNull()
          val y = it.groupValues[2].toDoubleOrNull()
          if (x != null && y != null) x to y else null
        },
      strictJson = false,
    )
  }

  private fun pair(a: JSONArray?): Pair<Double, Double>? {
    if (a == null || a.length() != 2) return null
    val x = a.optDouble(0)
    val y = a.optDouble(1)
    return if (x.isNaN() || y.isNaN()) null else x to y
  }

  private fun strings(a: JSONArray?): List<String> =
    if (a == null) emptyList() else (0 until a.length()).map { a.optString(it) }

  /**
   * The model's 0-999 point as pixels of a [width] x [height] screenshot, the way the vendor's agent
   * converts it: round(x / 999 * (width - 1)).
   */
  fun toPixels(point: Pair<Double, Double>, width: Int, height: Int): Pair<Int, Int> {
    val x = point.first.coerceIn(0.0, 999.0)
    val y = point.second.coerceIn(0.0, 999.0)
    return pyRound(x / 999.0 * (width - 1)) to pyRound(y / 999.0 * (height - 1))
  }

  /** Python's round(): a value halfway between two integers goes to the even one. */
  private fun pyRound(v: Double): Int = Math.rint(v).toInt()
}
