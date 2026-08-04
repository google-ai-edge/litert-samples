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

package com.google.ai.edge.examples.text_to_speech_streaming

/**
 * Splits input text into per-sentence synthesis chunks — a faithful port of the upstream pip
 * package's `chunk_text`, which is also the streaming granularity: each chunk is synthesized and
 * queued for playback while the next one computes, so time-to-first-audio is one short sentence.
 *
 * Upstream quirks kept on purpose (the model's prosody was tuned against this frontend):
 * sentence-final `.!?` are consumed by the split, and every chunk is then terminated with a comma,
 * so the model always sees `,` as the chunk-final token.
 */
object SentenceChunker {

  private val SENTENCE_END = Regex("[.!?]+")
  private const val MAX_CHUNK_CHARS = 400
  private const val PUNCTUATION = ".!?,;:"

  fun chunk(text: String): List<String> {
    val chunks = ArrayList<String>()
    for (sentence in SENTENCE_END.split(text)) {
      val trimmed = sentence.trim()
      if (trimmed.isEmpty()) continue
      if (trimmed.length <= MAX_CHUNK_CHARS) {
        chunks.add(ensurePunctuation(trimmed))
      } else {
        // Overlong sentence: split on word boundaries.
        val builder = StringBuilder()
        for (word in trimmed.split(Regex("\\s+"))) {
          if (builder.length + word.length + 1 > MAX_CHUNK_CHARS && builder.isNotEmpty()) {
            chunks.add(ensurePunctuation(builder.toString()))
            builder.setLength(0)
          }
          if (builder.isNotEmpty()) builder.append(' ')
          builder.append(word)
        }
        if (builder.isNotEmpty()) chunks.add(ensurePunctuation(builder.toString()))
      }
    }
    return chunks
  }

  private fun ensurePunctuation(sentence: String): String =
    if (sentence.last() in PUNCTUATION) sentence else "$sentence,"
}
