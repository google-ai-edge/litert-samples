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
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/Markup.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

/**
 * The index of [opener] in [text], or of a trailing piece of it that the next chunk may complete;
 * the text's length when neither is there. (Its own file: the JVM tests load it without LiteRT-LM's
 * classes.)
 */
internal fun beforeMarkup(text: String, opener: String): Int {
  val at = text.indexOf(opener)
  if (at >= 0) {
    return at
  }
  for (k in minOf(opener.length - 1, text.length) downTo 1) {
    if (text.regionMatches(text.length - k, opener, 0, k)) {
      return text.length - k
    }
  }
  return text.length
}

/**
 * The openers of the call markup the runtime parses for the model types it knows (Qwen's
 * `<tool_call>`, Gemma 4's `<|tool_call>`, FunctionGemma's `<start_function_call>`): in the text of
 * a turn any of them is a call the runtime left unparsed.
 */
internal val RUNTIME_MARKUP = listOf("<tool_call", "<|tool_call", "<start_function_call>")

/**
 * The part of [said] (a finished turn's text without call markup, trimmed) that the first [shown]
 * characters of [text] (the raw turn, streamed up to the first markup) did not show yet; empty when
 * nothing is left.
 */
internal fun unshownText(text: String, shown: Int, said: String): String {
  val before = text.substring(0, shown).trimStart()
  return if (said.length > before.length && said.startsWith(before)) {
    said.substring(before.length)
  } else {
    ""
  }
}

private val LINE_MARK = Regex("(?m)^[ \\t]*(?:[-*] +|#+[ \\t]*)")
private val MARKDOWN_MARKS = Regex("[*_#`]")

/**
 * [text] without the markdown a chat model may still write in a spoken reply: a `- ` or `* `
 * bullet or a `#` heading mark at the start of a line, and every `*`, `_`, `#` and `` ` ``
 * (emphasis, code). What is left goes to the speaker; the model's text as it was written stays in
 * [VoiceLoop.TurnTiming.reply].
 */
internal fun stripMarkdown(text: String): String =
  text.replace(LINE_MARK, "").replace(MARKDOWN_MARKS, "")
