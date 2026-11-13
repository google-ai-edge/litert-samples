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

package com.google.ai.edge.examples.image_segmentation

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.examples.image_segmentation.ImageSegmentationHelper.AcceleratorEnum
import com.google.ai.edge.examples.image_segmentation.ImageSegmentationHelper.SegmentationResult
import java.util.concurrent.CopyOnWriteArrayList
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumentation tests for [ImageSegmentationHelper], running on the LiteRT runtime provided by
 * Google Play services.
 *
 * They run on a device or emulator with Google Play services, and check the first things a
 * developer trying this sample relies on: LiteRT initializes through Google Play services, the
 * model compiles, and segmenting an image returns a mask with one label per pixel.
 *
 * Run them with `./gradlew :app:connectedDebugAndroidTest`.
 */
@RunWith(AndroidJUnit4::class)
class ImageSegmentationHelperTest {
  private lateinit var helper: ImageSegmentationHelper

  // Errors reported by the helper. The helper never throws: it reports failures through its
  // `error` flow, so the tests collect them to explain why a test failed.
  private val reportedErrors = CopyOnWriteArrayList<Throwable?>()
  private lateinit var errorCollector: Job

  @Before
  fun setUp() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    helper = ImageSegmentationHelper(context)

    // `error` is a SharedFlow without replay, so errors emitted before anyone collects are lost.
    // CoroutineStart.UNDISPATCHED subscribes right away, before any test calls the helper.
    errorCollector =
      CoroutineScope(Dispatchers.Default).launch(start = CoroutineStart.UNDISPATCHED) {
        helper.error.collect { error ->
          Log.i(TAG, "The helper reported an error", error)
          reportedErrors.add(error)
        }
      }
  }

  @After
  fun tearDown() {
    errorCollector.cancel()
    // Releases the segmenter and the LiteRT environment from Google Play services.
    helper.close()
  }

  @Test
  fun segment_onCpu_returnsOneLabelPerPixel() = runBlocking {
    // Initializes LiteRT from Google Play services and compiles the model for the CPU
    initSegmenterOrFail(AcceleratorEnum.CPU)

    val result = segmentTestImage()

    assertHasOneLabelPerPixel(result)
    // The CPU is always available, so the helper must not report any error.
    assertTrue(
      "Unexpected errors reported by the helper: $reportedErrors",
      reportedErrors.isEmpty(),
    )
  }

  @Test
  fun segment_onGpu_returnsOneLabelPerPixelOnGpuOrCpuFallback() = runBlocking {
    // Requests the GPU. Emulators have no GPU that LiteRT supports: in that case the helper reports
    // an IllegalStateException and falls back to the CPU.
    initSegmenterOrFail(AcceleratorEnum.GPU)

    val result = segmentTestImage()

    assertHasOneLabelPerPixel(result)
    // On emulators, the only acceptable error is the fallback to the CPU; any other
    // error is a real failure, even if the CPU fallback then produced a result.
    for (error in reportedErrors) {
      assertTrue(
        "Unexpected error reported by the helper: $error",
        error is IllegalStateException && error.message?.contains(CPU_FALLBACK_MESSAGE) == true,
      )
    }
  }

  /**
   * Initializes the helper's segmenter for [accelerator], and fails the test if that doesn't finish
   * within [INIT_TIMEOUT_MS].
   */
  private suspend fun initSegmenterOrFail(accelerator: AcceleratorEnum) {
    val initialized = withTimeoutOrNull(INIT_TIMEOUT_MS) { helper.initSegmenter(accelerator) }
    assertNotNull(
      "initSegmenter($accelerator) didn't finish within $INIT_TIMEOUT_MS ms." +
        " Errors reported by the helper: $reportedErrors",
      initialized,
    )
  }

  /**
   * Checks that [result] holds one 256x256 mask in which every pixel is labeled with one of the
   * model's classes. On failure, the message lists the errors the helper reported, which usually
   * explain why there is no result (e.g. Google Play services could not load LiteRT).
   */
  private fun assertHasOneLabelPerPixel(result: SegmentationResult?) {
    val nonNullResult =
      checkNotNull(result) {
        "No segmentation result. Errors reported by the helper: $reportedErrors"
      }
    val masks = nonNullResult.segmentation.masks
    assertEquals(1, masks.size)
    val mask = masks[0]
    assertEquals(MODEL_OUTPUT_SIZE, mask.width)
    assertEquals(MODEL_OUTPUT_SIZE, mask.height)
    assertEquals(MODEL_OUTPUT_SIZE * MODEL_OUTPUT_SIZE, mask.data.capacity())
    for (pixel in 0 until mask.data.capacity()) {
      val label = mask.data.get(pixel).toInt()
      assertTrue("Pixel $pixel has label $label", label in 0 until MODEL_OUTPUT_CLASSES)
    }
  }

  /**
   * Segments a synthetic test image and returns the first result, or null if the helper doesn't
   * produce one within [RESULT_TIMEOUT_MS] (e.g. because initialization failed and the helper has
   * no segmenter).
   */
  private suspend fun segmentTestImage(): SegmentationResult? = coroutineScope {
    // `segmentation` is also a SharedFlow without replay, so subscribe (UNDISPATCHED) before
    // asking the helper to segment, or the result would be lost.
    val firstResult = async(start = CoroutineStart.UNDISPATCHED) { helper.segmentation.first() }

    // The content of the image doesn't matter for these tests: they check that the model runs and
    // that every pixel gets a valid label. A non-square size also exercises the resize step.
    val image = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
    image.eraseColor(Color.rgb(200, 120, 80))
    helper.segment(image, /* rotationDegrees= */ 0)

    val result = withTimeoutOrNull(RESULT_TIMEOUT_MS) { firstResult.await() }
    firstResult.cancel()
    result
  }

  companion object {
    private const val TAG = "ImageSegmentationTest"

    // The selfie multiclass model returns a 256x256 mask with 6 classes (see
    // ImageSegmentationHelper), so every pixel's label is a class index in [0, 6).
    private const val MODEL_OUTPUT_SIZE = 256
    private const val MODEL_OUTPUT_CLASSES = 6

    // Segmenting one image takes well under a second on a device; emulators are much slower.
    private const val RESULT_TIMEOUT_MS = 60_000L

    // Initializing LiteRT through Google Play services the first time can take much longer than
    // one segmentation, especially on emulators.
    private const val INIT_TIMEOUT_MS = 120_000L

    // Part of the message of the IllegalStateException that ImageSegmentationHelper reports when it
    // can't use the requested accelerator and falls back to the CPU.
    private const val CPU_FALLBACK_MESSAGE = "using CPU instead"
  }
}
