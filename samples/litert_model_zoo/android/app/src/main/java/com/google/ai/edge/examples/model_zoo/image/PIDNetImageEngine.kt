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

package com.google.ai.edge.examples.model_zoo.image

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import com.google.ai.edge.examples.model_zoo.models.pidnet.Segmenter
import java.io.File

class PIDNetImageEngine(context: Context, modelDir: File, preferredBackend: String) :
  SingleImageEngine {
  private val compiled =
    compileImageBackend(preferredBackend, "PIDNetImageEngine") { Segmenter(context, modelDir, it) }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val (label, ms) = compiled.runner.segment(request.bitmap)
    val output = request.bitmap.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(output)
    val b = label
    val width = output.width
    val height = output.height
    val src = Rect()
    val dst = Rect()
    val paint = Paint(Paint.FILTER_BITMAP_FLAG).apply { alpha = 130 }
    src.set(0, 0, b.width, b.height)
    dst.set(0, 0, width, height)
    canvas.drawBitmap(b, src, dst, paint)
    return ImageTaskOutput(
      bitmap = output,
      text =
        "Cityscapes segmentation: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle",
      inferenceMs = ms.toDouble(),
      backend = compiled.backend,
      fallbackReason = compiled.fallbackReason,
    )
  }

  override fun close() = compiled.runner.close()
}
