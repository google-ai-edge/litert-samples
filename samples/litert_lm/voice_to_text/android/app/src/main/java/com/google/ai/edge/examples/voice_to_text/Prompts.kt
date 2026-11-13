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

import kotlin.math.ceil

/** What the model is asked to do with a recording: the prompts of the model card. */
enum class Task(val label: String) {
  /** The words in the language spoken, with punctuation and capitals. */
  TRANSCRIPT("Transcript"),

  /** The speech translated into English. */
  ENGLISH("English translation"),
}

/**
 * The instruction that follows the audio in the message, from the source model card's table of
 * prompts (ibm-granite/granite-speech-5.0-2b, "Preferred prompt by task"), and the output budget.
 */
object Prompts {
  /** Characters of keyword list the screen accepts. */
  const val MAX_KEYWORD_CHARS = 500

  /**
   * The recording cap. The model card's 2,048-token context holds about 116 s of audio with the
   * longest reply of its 220 test clips (10 audio positions per second); 90 s leaves room.
   */
  const val MAX_SECONDS = 90

  /** The context of `granite-speech-5.0-2b_ekv2048.litertlm`: prompt and reply together. */
  const val CONTEXT_TOKENS = 2048

  /**
   * The keyword list as the prompt writes it: comma- or line-separated words, trimmed, empty ones
   * and repeats dropped, in the order typed.
   */
  fun keywords(text: String): List<String> =
    text.take(MAX_KEYWORD_CHARS).split(',', '\n', ';').map { it.trim() }.filter { it.isNotEmpty() }.distinct()

  fun prompt(task: Task, keywords: List<String>): String {
    val list = keywords.joinToString(", ")
    return when (task) {
      Task.TRANSCRIPT ->
        if (keywords.isEmpty()) "transcribe the speech with proper punctuation and capitalization."
        else "transcribe the speech to text. Keywords: $list"
      Task.ENGLISH ->
        if (keywords.isEmpty())
          "translate the speech to English with proper punctuation and capitalization."
        else "translate the speech to English. Keywords: $list"
    }
  }

  /**
   * The most tokens a reply may take: 10 per second of audio plus 100. The longest reply among the
   * model card's 220 test clips was about 7.5 tokens per second of audio, so this cap only ends a
   * reply that repeats itself. It never asks for more than the context has left after the audio
   * (10 positions per second) and the instruction.
   */
  fun maxOutputTokens(seconds: Double, prompt: String): Int {
    val audio = ceil(seconds * 10).toInt()
    // Generous for the instruction and the chat template's own text.
    val instruction = 24 + prompt.length / 2
    val left = CONTEXT_TOKENS - audio - instruction
    val wanted = ceil(seconds * 10).toInt() + 100
    return maxOf(16, minOf(wanted, left))
  }
}
