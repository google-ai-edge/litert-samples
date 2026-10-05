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
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/ToolFormat.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.ToolCall
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.tool
import org.json.JSONObject

/**
 * How the tool calls travel: LiteRT-LM parses Gemma 4's calls itself (the bundle's
 * `LlmMetadata.llm_model_type` is gemma4). The tools go in `ConversationConfig.tools`, the calls
 * come back in `Message.toolCalls`, and the app runs them (`automaticToolCalling = false`). Call
 * markup left in the text ([RUNTIME_MARKUP]: a call the runtime did not parse) is never shown and
 * fails the turn instead of being said.
 */
internal object RuntimeToolFormat {
  /** The tools as the runtime declares them to the model. */
  fun providers(tools: List<VoiceTool>): List<ToolProvider> =
    tools.map { tool(OpenApiToolAdapter(it)) }

  /** A finished model turn: the runtime's calls, or why the turn cannot go on. */
  fun parse(text: String, runtimeCalls: List<ToolCall>): ParsedTurn {
    val at = RUNTIME_MARKUP.map { text.indexOf(it) }.filter { it >= 0 }.minOrNull()
    if (at != null) {
      val markup = text.substring(at).take(80)
      return ParsedTurn(
        emptyList(),
        text.trim(),
        "tool call markup the runtime did not parse is in the text: $markup",
      )
    }
    return ParsedTurn(runtimeCalls.map { ParsedCall(it.name, it.arguments) }, text.trim(), null)
  }

  /**
   * The answer to one call. The runtime's FC-format processor (Gemma 4) renders an object response
   * as `name{result:...}` and would print a bare string's JSON wrapper fields too, so the answer
   * is `{"result": text}`.
   */
  fun response(name: String, result: String): Content =
    Content.ToolResponse(name, mapOf("result" to result))

  /**
   * How much of a turn's streamed text can be shown (or spoken) now: everything before the first
   * call markup, holding back an end that may be the start of it. The runtime keeps its own calls
   * out of the text.
   */
  fun visibleLength(text: String): Int = RUNTIME_MARKUP.minOf { beforeMarkup(text, it) }
}

/** One call to run. */
internal data class ParsedCall(val name: String, val args: Map<String, Any?>)

/** A finished model turn: the calls to run, the text without them, or why the turn cannot go on. */
internal data class ParsedTurn(val calls: List<ParsedCall>, val said: String, val error: String?)

/**
 * A [VoiceTool] declared to LiteRT-LM. The runtime never runs it: the conversation passes
 * `automaticToolCalling = false`, and [ToolRunner] runs the calls itself, so [execute] being called
 * is a bug.
 */
internal class OpenApiToolAdapter(private val tool: VoiceTool) : OpenApiTool {
  override fun getToolDescriptionJsonString(): String = JSONObject(tool.functionMap()).toString()

  override fun execute(paramsJsonString: String): String =
    throw IllegalStateException(
      "the runtime called ${tool.name} itself; ToolRunner runs the tools " +
        "(automaticToolCalling is off)"
    )
}
