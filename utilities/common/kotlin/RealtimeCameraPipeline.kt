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

package __MODULE_PACKAGE__
// Vendored from common/kotlin/RealtimeCameraPipeline.kt — edit the canonical and run tools/sync_common.py --apply.

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Reusable two-thread realtime camera + ML inference pipeline.
 *
 * Architecture:
 *   Camera thread:    proxyToBitmap → frameChannel
 *   Inference thread: poll frame → onFrame callback → return bitmap to free pool
 *
 * Benefits over single-threaded pipeline:
 *   - bmp conversion is hidden behind GPU inference time
 *   - Always processes the latest available frame (drops stale ones)
 *   - Bitmap pool eliminates per-frame allocation
 *
 * Usage:
 *   val pipeline = RealtimeCameraPipeline(activity, previewView) { bmp ->
 *       val results = detector.detect(bmp)
 *       runOnUiThread { overlayView.setResults(results, bmp.width, bmp.height) }
 *   }
 *   pipeline.start(this)  // call from onCreate
 *   // ...
 *   pipeline.close()  // call from onDestroy
 *
 * The [onFrame] callback runs on the inference thread. Post UI updates with
 * `runOnUiThread { ... }` or `view.post { ... }`.
 *
 * To gate inference (e.g. while loading models), set [enabled] to false.
 */
class RealtimeCameraPipeline(
    private val activity: androidx.activity.ComponentActivity,
    private val previewView: PreviewView? = null,
    private val analysisResolution: Size = Size(384, 288),
    private val previewResolution: Size = Size(640, 480),
    private val cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA,
    private val onFrame: (Bitmap) -> Unit,
) : AutoCloseable {

    companion object {
        private const val TAG = "CameraPipeline"
    }

    @Volatile
    var enabled: Boolean = true

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val inferenceExecutor = Executors.newSingleThreadExecutor()

    private val freePool = ArrayBlockingQueue<Bitmap>(2)
    private val frameChannel = ArrayBlockingQueue<Bitmap>(1)

    @Volatile
    private var inferenceRunning = false

    // Camera-thread local state
    private val mat = Matrix()
    private val pnt = Paint(Paint.FILTER_BITMAP_FLAG)
    private var srcBitmap: Bitmap? = null
    private var canvas1Bmp: Bitmap? = null
    private var canvas1: Canvas? = null
    private var canvas2Bmp: Bitmap? = null
    private var canvas2: Canvas? = null

    // FPS tracking (inference thread → main thread reads)
    @Volatile
    var fps: Int = 0
        private set
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()

    /**
     * Start the camera pipeline. Call from onCreate after layout is ready.
     * The inference loop starts immediately and runs until [close].
     */
    fun start(lifecycleOwner: LifecycleOwner) {
        startInferenceLoop()
        bindCamera(lifecycleOwner)
    }

    private fun startInferenceLoop() {
        inferenceRunning = true
        inferenceExecutor.execute {
            while (inferenceRunning) {
                val bmp = try {
                    frameChannel.poll(100, TimeUnit.MILLISECONDS)
                } catch (e: InterruptedException) {
                    break
                } ?: continue

                try {
                    if (bmp.isRecycled) continue
                    onFrame(bmp)

                    frameCount++
                    val now = System.currentTimeMillis()
                    if (now - lastFpsTime >= 1000) {
                        fps = frameCount
                        frameCount = 0
                        lastFpsTime = now
                    }
                } catch (e: Throwable) {
                    Log.e(TAG, "Inference error: ${e.message}", e)
                } finally {
                    if (!bmp.isRecycled) freePool.offer(bmp)
                }
            }
        }
    }

    private fun bindCamera(lifecycleOwner: LifecycleOwner) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(activity)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            @Suppress("DEPRECATION")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(analysisResolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(cameraExecutor) { proxy ->
                if (!enabled) {
                    proxy.close()
                    return@setAnalyzer
                }
                try {
                    handleProxy(proxy)
                } catch (e: Throwable) {
                    Log.e(TAG, "Camera thread error: ${e.message}", e)
                    try { proxy.close() } catch (_: Exception) {}
                }
            }

            cameraProvider.unbindAll()
            if (previewView != null) {
                @Suppress("DEPRECATION")
                val preview = Preview.Builder()
                    .setTargetResolution(previewResolution)
                    .build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }
                cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, analysis)
            } else {
                cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, analysis)
            }
        }, ContextCompat.getMainExecutor(activity))
    }

    private fun handleProxy(proxy: ImageProxy) {
        // Lazy init pool on first frame (need rotation info)
        if (canvas1Bmp == null && canvas2Bmp == null) {
            val rot = proxy.imageInfo.rotationDegrees
            val w = if (rot == 90 || rot == 270) proxy.height else proxy.width
            val h = if (rot == 90 || rot == 270) proxy.width else proxy.height
            freePool.offer(Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888))
            freePool.offer(Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888))
        }

        // Get a bitmap to write to (prefer free, fall back to displacing stale)
        val bmp = freePool.poll() ?: frameChannel.poll() ?: run {
            proxy.close()
            return
        }

        proxyToBitmapInto(proxy, bmp)
        proxy.close()

        // Replace any frame currently in channel with the new (latest) one
        val displaced = frameChannel.poll()
        if (!frameChannel.offer(bmp)) {
            freePool.offer(bmp)
        }
        if (displaced != null) freePool.offer(displaced)
    }

    /**
     * Convert ImageProxy into the supplied destination bitmap (camera-thread only).
     * Uses pre-allocated src bitmap and per-dst Canvas cache to avoid allocations.
     */
    private fun proxyToBitmapInto(proxy: ImageProxy, dst: Bitmap) {
        val plane = proxy.planes[0]
        val buf = plane.buffer
        val sw = proxy.width + (plane.rowStride - plane.pixelStride * proxy.width) / plane.pixelStride

        var src = srcBitmap
        if (src == null || src.width != sw || src.height != proxy.height) {
            src?.recycle()
            src = Bitmap.createBitmap(sw, proxy.height, Bitmap.Config.ARGB_8888)
            srcBitmap = src
        }
        buf.rewind()
        src.copyPixelsFromBuffer(buf)

        val rot = proxy.imageInfo.rotationDegrees.toFloat()
        val canvas = canvasFor(dst)

        if (rot == 0f && sw == proxy.width) {
            canvas.drawBitmap(src, 0f, 0f, pnt)
            return
        }

        val rw = if (rot == 90f || rot == 270f) proxy.height else proxy.width
        val rh = if (rot == 90f || rot == 270f) proxy.width else proxy.height
        mat.reset()
        mat.postRotate(rot, proxy.width / 2f, proxy.height / 2f)
        mat.postTranslate((rw - proxy.width) / 2f, (rh - proxy.height) / 2f)
        canvas.drawBitmap(src, mat, pnt)
    }

    private fun canvasFor(bmp: Bitmap): Canvas {
        if (canvas1Bmp === bmp) return canvas1!!
        if (canvas2Bmp === bmp) return canvas2!!
        return if (canvas1Bmp == null) {
            canvas1Bmp = bmp
            Canvas(bmp).also { canvas1 = it }
        } else {
            canvas2Bmp = bmp
            Canvas(bmp).also { canvas2 = it }
        }
    }

    override fun close() {
        inferenceRunning = false
        cameraExecutor.shutdown()
        inferenceExecutor.shutdown()
        srcBitmap?.recycle(); srcBitmap = null
        while (true) freePool.poll()?.recycle() ?: break
        while (true) frameChannel.poll()?.recycle() ?: break
        canvas1 = null; canvas1Bmp = null
        canvas2 = null; canvas2Bmp = null
    }
}
