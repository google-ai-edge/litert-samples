/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.asr

import android.content.Context

/** Tokenizer implementation with native Huggingface tokenizer. */
class HuggingfaceTokenizer(context: Context, modelConfig: ModelConfig) : Tokenizer {
  private val nativeHandle: Long
  override val vocabSize: Int

  init {
    val jsonString =
      readOrDownloadFile(context, modelConfig.tokenizerPath, modelConfig.tokenizerRemoteUrl)
    nativeHandle = nativeInit(jsonString)
    if (nativeHandle == 0L) {
      throw IllegalStateException("Failed to initialize native Tokenizer.")
    }

    vocabSize = nativeGetVocabSize(nativeHandle)
  }

  override fun close() {
    nativeFree(nativeHandle)
  }

  override fun decode(tokenIds: IntArray): String =
    nativeDecode(nativeHandle, tokenIds, skipSpecialTokens = true).trim()

  private companion object {
    init {
      System.loadLibrary("tokenizer_jni")
    }

    external fun nativeInit(jsonPayload: String): Long

    external fun nativeFree(handle: Long)

    external fun nativeDecode(handle: Long, ids: IntArray, skipSpecialTokens: Boolean): String

    external fun nativeGetVocabSize(handle: Long): Int
  }
}
