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

package com.google.ai.edge.examples.ocr_verification

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.io.Closeable

/** One line the text recognizer read: its text, its box in the photo and its own confidence. */
class TextLine(val text: String, val box: Rect, val confidence: Float)

/**
 * The text lines of a photo from ML Kit's on-device text recognizer (Latin script, the model
 * bundled in the app: no network), top to bottom.
 */
class TextRegions : Closeable {
  private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

  fun lines(bitmap: Bitmap, done: (Result<List<TextLine>>) -> Unit) {
    recognizer
      .process(InputImage.fromBitmap(bitmap, 0))
      .addOnSuccessListener { text ->
        val out = ArrayList<TextLine>()
        for (block in text.textBlocks) {
          for (line in block.lines) {
            val box = line.boundingBox ?: continue
            if (line.text.isBlank()) continue
            out.add(TextLine(line.text, box, line.confidence))
          }
        }
        // reading order of a one-column page: top to bottom, then left to right
        out.sortWith(compareBy<TextLine>({ it.box.top }, { it.box.left }))
        done(Result.success(out))
      }
      .addOnFailureListener { done(Result.failure(it)) }
  }

  override fun close() = recognizer.close()
}
