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

/** The sentences the screen shows when something does not work, and what to do about it. */
object Messages {
  const val NO_MODEL = "Download the model first: tap Download (2.8 GB, once)."

  const val PERMISSION_DENIED =
    "Voice to Text needs the microphone to hear you. Tap Record and allow it."

  const val PERMISSION_BLOCKED =
    "Microphone access is off for Voice to Text. Turn it on in the app's settings: " +
      "Permissions > Microphone."

  fun capReached(seconds: Int): String =
    "Recording stopped at %d:%02d, the most this sample sends in one message. Record the rest as a new message."
      .format(seconds / 60, seconds % 60)

  const val BACKGROUND =
    "Recording stopped because the app left the screen. What was recorded is transcribed."

  const val CANCELLED = "Stopped. Nothing more will be written for this recording."

  /** A runtime refusal or failure while transcribing, as a sentence. */
  fun transcribeFailed(t: Throwable): String {
    if (t is EngineUnavailableException) return t.message ?: "The model is not loaded."
    val raw = t.message ?: t.javaClass.simpleName
    return if (isTooLong(raw)) {
      "This recording is too long for the model: it holds 2,048 tokens for the audio and its " +
        "text together (about 90 s of speech). Record a shorter piece."
    } else {
      "The model could not transcribe this recording ($raw)."
    }
  }

  /**
   * True for the runtime's refusal of a message that does not fit in the context: status 3
   * (INVALID_ARGUMENT) with words about tokens or length.
   */
  fun isTooLong(raw: String): Boolean {
    val text = raw.lowercase()
    val invalidArgument = text.contains("status code: 3.") || text.contains("invalid_argument")
    val aboutLength =
      listOf("token", "exceed", "too long", "max_num", "context", "length").any { text.contains(it) }
    return invalidArgument && aboutLength
  }

  /** Making the Engine failed, as a sentence. */
  fun engineFailed(t: Throwable): String =
    when (t) {
      is NotEnoughMemoryException -> t.message ?: "Not enough free memory."
      else ->
        "The model did not start on this phone's GPU (${t.message ?: t.javaClass.simpleName}). " +
          "Tap Load to try again."
    }

  /** A download that stopped, as a sentence that says the received part is kept. */
  fun downloadFailed(t: Throwable): String {
    val raw = t.message ?: t.javaClass.simpleName
    return if (raw.contains("SHA-256") || raw.contains("verification")) {
      "The downloaded file did not match its checksum and was deleted. Tap Download to get it again."
    } else {
      "The download stopped ($raw). The part already downloaded is kept: tap Download to resume."
    }
  }
}
