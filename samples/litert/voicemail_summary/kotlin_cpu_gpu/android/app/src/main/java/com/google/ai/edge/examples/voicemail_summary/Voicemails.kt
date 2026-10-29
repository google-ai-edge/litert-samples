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

package com.google.ai.edge.examples.voicemail_summary

import android.content.Context
import org.json.JSONObject

/**
 * The example voicemails in the app's assets (`voicemails/<file>`, 16 kHz mono WAV, listed in `voicemails.json`):
 * English, written for this sample, with fictional names and 555-01xx numbers, spoken by Kokoro-82M (README.md).
 * [caller] and [callback] are only for the measuring tool's checks; the screen never shows them before an answer.
 */
class BundledVoicemail(val id: String, val label: String, val file: String, val caller: String, val callback: String) {
  fun load(context: Context): Voicemail {
    val pcm = context.assets.open("voicemails/$file").use { AudioIO.readWav16k(it.readBytes()) }
    return AudioIO.check(label, pcm, pcm.size / 16000.0)
  }

  companion object {
    fun list(context: Context): List<BundledVoicemail> {
      val items = JSONObject(context.assets.open("voicemails.json").bufferedReader().readText()).getJSONArray("items")
      return List(items.length()) { i ->
        val o = items.getJSONObject(i)
        BundledVoicemail(o.getString("id"), o.getString("label"), o.getString("file"), o.getString("caller"), o.getString("callback"))
      }
    }
  }
}
