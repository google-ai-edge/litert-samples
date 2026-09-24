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
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.google.ai.edge.examples.model_zoo.models.crowdcount.CrowdCounter
import com.google.ai.edge.examples.model_zoo.models.dehaze.Dehazer
import com.google.ai.edge.examples.model_zoo.models.dinov2.Dinov2Features
import com.google.ai.edge.examples.model_zoo.models.modnet.Matter
import com.google.ai.edge.examples.model_zoo.models.nima.NimaScorer
import com.google.ai.edge.examples.model_zoo.models.plantnet.PlantClassifier
import com.google.ai.edge.examples.model_zoo.models.vrwkv.VrwkvClassifier
import com.google.ai.edge.examples.model_zoo.models.xfeat.XFeatMatcher
import java.io.File

/** Catalog-backed adapters: numerical inference stays in the model wrappers. */
object RemainingImageTasks {
  val ids =
    setOf(
      "image-dehazing",
      "portrait-matting",
      "image-quality",
      "fine-grained-classification",
      "image-classification",
      "crowd-counting",
      "dense-feature-visualization",
      "image-matching",
    )

  fun create(
    taskId: String,
    context: Context,
    directory: File,
    backend: String,
  ): SingleImageEngine =
    when (taskId) {
      "image-dehazing" ->
        engine(
          compileImageBackend(backend, "ModelZooDehaze") {
            Dehazer(context, directory.model("dehazeformer_base.tflite"), it)
          },
          { it.close() },
        ) { runner, request ->
          val (bitmap, _) = runner.dehaze(request.bitmap)
          Rendered(bitmap, "Dehazed image")
        }
      "portrait-matting" ->
        engine(
          compileImageBackend(backend, "ModelZooModnet") {
            Matter(context, directory.model("modnet.tflite"), it)
          },
          { it.close() },
        ) { runner, request ->
          val (bitmap, _) = runner.matte(request.bitmap, Color.rgb(30, 190, 120))
          Rendered(
            checkNotNull(bitmap.copy(Bitmap.Config.ARGB_8888, false)),
            "Portrait on studio green",
          )
        }
      "image-quality" ->
        engine(
          compileImageBackend(backend, "ModelZooNima") {
            NimaScorer(context, directory.model("nima_aesthetic_fp16.tflite"), null, it)
          },
          { it.close() },
        ) { runner, request ->
          val scores = runner.score(request.bitmap)
          Rendered(
            text =
              "Aesthetic score: %.2f / 10\n%s"
                .format(
                  scores.aesthetic,
                  scores.distribution
                    .mapIndexed { i, probability -> "${i + 1}: %.1f%%".format(probability * 100) }
                    .joinToString(" · "),
                ),
            metrics =
              mapOf(
                "meanScore" to scores.aesthetic,
                "distribution" to scores.distribution.toList(),
                "technicalScore" to scores.technical,
              ),
          )
        }
      "fine-grained-classification" ->
        engine(
          compileImageBackend(backend, "ModelZooPlantnet") {
            PlantClassifier(
              context,
              directory.model("plantnet.tflite"),
              directory.model("plantnet300K_species_id_2_name.json"),
              it,
            )
          },
          { it.close() },
        ) { runner, request ->
          val (predictions, _) = runner.classify(request.bitmap, 5)
          Rendered(
            text =
              predictions.joinToString("\n") { (name, score) ->
                "%s — %.1f%%".format(name, score * 100)
              },
            metrics =
              mapOf(
                "top5" to
                  predictions.map { (name, score) -> mapOf("label" to name, "score" to score) }
              ),
          )
        }
      "image-classification" ->
        engine(
          compileImageBackend(backend, "ModelZooVrwkv") {
            VrwkvClassifier(
              context,
              directory.model("vrwkv_s_fp16.tflite"),
              directory.model("imagenet_classes.txt"),
              it,
            )
          },
          { it.close() },
        ) { runner, request ->
          val predictions = runner.classify(request.bitmap, 5)
          Rendered(
            text =
              predictions.joinToString("\n") {
                "%s — %.1f%%".format(it.label, it.probability * 100)
              },
            metrics =
              mapOf(
                "top5" to predictions.map { mapOf("label" to it.label, "score" to it.probability) }
              ),
          )
        }
      "crowd-counting" ->
        engine(
          compileImageBackend(backend, "ModelZooCrowd") {
            CrowdCounter(context, directory.model("dmcount.tflite"), it)
          },
          { it.close() },
        ) { runner, request ->
          val result = runner.count(request.bitmap)
          Rendered(
            densityOverlay(request.bitmap, result),
            "Estimated people: %.2f".format(result.count),
            mapOf(
              "count" to result.count,
              "densityShape" to listOf(1, 1, CrowdCounter.OUT, CrowdCounter.OUT),
            ),
          )
        }
      "dense-feature-visualization" ->
        engine(
          compileImageBackend(backend, "ModelZooDinov2") {
            Dinov2Features(context, directory.model("dinov2_s_fp16.tflite"), it)
          },
          { it.close() },
        ) { runner, request ->
          val features = runner.featureMap(request.bitmap)
          val display =
            Bitmap.createScaledBitmap(features, Dinov2Features.SIZE, Dinov2Features.SIZE, false)
          if (display !== features) features.recycle()
          Rendered(
            display,
            "DINOv2 patch features · PCA to RGB",
            mapOf("patchGrid" to listOf(32, 32), "featureDimension" to 384),
          )
        }
      "image-matching" ->
        engine(
          compileImageBackend(backend, "ModelZooXfeat") {
            XFeatMatcher(context, directory.model("xfeat_fp16.tflite"), it)
          },
          { it.close() },
        ) { runner, request ->
          val second =
            requireNotNull(request.secondaryBitmap) { "Pick a second image to find matches." }
          val a = runner.extract(runner.preprocess(request.bitmap))
          val b = runner.extract(runner.preprocess(second))
          val matches = runner.match(a, b)
          Rendered(
            matchOverlay(request.bitmap, second, matches),
            com.google.ai.edge.examples.model_zoo.ResultCounts.matches(matches.size),
            mapOf(
              "matchCount" to matches.size,
              "keypointsFirst" to a.xs.size,
              "keypointsSecond" to b.xs.size,
              "first10Matches" to
                matches.take(10).map {
                  mapOf(
                    "x0" to it.x0,
                    "y0" to it.y0,
                    "x1" to it.x1,
                    "y1" to it.y1,
                    "similarity" to it.sim,
                  )
                },
            ),
          )
        }
      else -> error("Unknown image task: $taskId")
    }

  private data class Rendered(
    val bitmap: Bitmap? = null,
    val text: String,
    val metrics: Map<String, Any?> = emptyMap(),
  )

  private fun <T> engine(
    loaded: ImageBackend<T>,
    close: (T) -> Unit,
    run: (T, ImageTaskRequest) -> Rendered,
  ): SingleImageEngine =
    object : SingleImageEngine {
      override fun run(request: ImageTaskRequest): ImageTaskOutput {
        val started = System.nanoTime()
        val result = run(loaded.runner, request)
        return ImageTaskOutput(
          result.bitmap,
          result.text,
          (System.nanoTime() - started) / 1e6,
          loaded.backend,
          loaded.fallbackReason,
          metrics = result.metrics,
        )
      }

      override fun close() = close(loaded.runner)
    }

  private fun File.model(name: String) =
    File(this, name).also { check(it.isFile) { "Download this task first: $name" } }

  // Renders the source and the density heatmap into an owned output bitmap.
  private fun densityOverlay(source: Bitmap, result: CrowdCounter.Result): Bitmap {
    val O = CrowdCounter.OUT
    val ovPixels = IntArray(O * O)
    var maxV = 1e-5f
    for (v in result.density) if (v > maxV) maxV = v
    for (i in 0 until O * O) {
      val v = (result.density[i] / maxV).coerceIn(0f, 1f)
      val a = (v * 220).toInt()
      val g = ((1f - v) * 160).toInt()
      ovPixels[i] = (a shl 24) or (0xFF shl 16) or (g shl 8)
    }
    val heat = Bitmap.createBitmap(ovPixels, O, O, Bitmap.Config.ARGB_8888)
    val output = Bitmap.createBitmap(CrowdCounter.SIZE, CrowdCounter.SIZE, Bitmap.Config.ARGB_8888)
    val bounds = RectF(0f, 0f, output.width.toFloat(), output.height.toFloat())
    val canvas = Canvas(output)
    canvas.drawBitmap(source, null, bounds, null)
    canvas.drawBitmap(heat, null, bounds, Paint(Paint.FILTER_BITMAP_FLAG))
    heat.recycle()
    return output
  }

  // Source-aspect display panels; model-coordinate matches are mapped for drawing.
  private fun matchOverlay(a: Bitmap, b: Bitmap, matches: List<XFeatMatcher.Match>): Bitmap {
    val height = XFeatMatcher.H
    val leftWidth = ImageDisplayGeometry.matchingPanelWidth(a.width, a.height, height)
    val rightWidth = ImageDisplayGeometry.matchingPanelWidth(b.width, b.height, height)
    val width = leftWidth + rightWidth
    val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val c = Canvas(output)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    val half = leftWidth.toFloat()
    val hgt = height.toFloat()
    c.drawBitmap(a, null, RectF(0f, 0f, half, hgt), null)
    c.drawBitmap(b, null, RectF(half, 0f, width.toFloat(), hgt), null)
    val sx = half / XFeatMatcher.W
    val rightSx = rightWidth.toFloat() / XFeatMatcher.W
    val sy = hgt / XFeatMatcher.H
    paint.strokeWidth = OverlaySizing.stroke(output.width)
    for (m in matches) {
      val t = ((m.sim - XFeatMatcher.MIN_COSSIM) / (1f - XFeatMatcher.MIN_COSSIM)).coerceIn(0f, 1f)
      paint.color = Color.argb(200, (255 * (1 - t)).toInt(), 220, 40)
      c.drawLine(m.x0 * sx, m.y0 * sy, half + m.x1 * rightSx, m.y1 * sy, paint)
      c.drawCircle(m.x0 * sx, m.y0 * sy, 3.5f, paint)
      c.drawCircle(half + m.x1 * rightSx, m.y1 * sy, 3.5f, paint)
    }
    return output
  }
}
