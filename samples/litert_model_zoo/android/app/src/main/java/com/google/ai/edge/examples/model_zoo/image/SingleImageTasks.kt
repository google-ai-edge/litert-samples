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
import java.io.File

/** Image tasks that have an engine; the catalog decides what can be downloaded. */
object SingleImageTasks {
  val ids =
    setOf(
      "background-removal",
      "super-resolution",
      "super-resolution-real-esrgan",
      "monocular-geometry-estimation",
      "semantic-segmentation",
      "instance-segmentation",
      "ocr",
      "image-tagging",
    ) + RealtimeImageTasks.ids + RemainingImageTasks.ids

  fun create(
    taskId: String,
    context: Context,
    directory: File,
    backend: String,
  ): SingleImageEngine =
    when (taskId) {
      "background-removal" -> OrmbgEngine(context, directory, backend)
      "super-resolution" -> EdsrEngine(context, directory, backend)
      "super-resolution-real-esrgan" -> RealEsrganEngine(context, directory, backend)
      "monocular-geometry-estimation" -> DA3ImageEngine(context, directory, backend)
      "semantic-segmentation" -> PIDNetImageEngine(context, directory, backend)
      "instance-segmentation" -> RfDetrSegImageEngine(context, directory, backend)
      "ocr" -> PpocrImageEngine(context, directory, backend)
      "image-tagging" -> RamImageEngine(context, directory, backend)
      in RealtimeImageTasks.ids -> RealtimeImageTasks.create(taskId, context, directory, backend)
      in RemainingImageTasks.ids -> RemainingImageTasks.create(taskId, context, directory, backend)
      else -> error("Unknown image task: $taskId")
    }
}
