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

package com.google.ai.edge.examples.litertruntime

import android.content.res.AssetManager
import android.os.Build
import android.os.Debug
import android.os.Process
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Measures what one model costs the process for each way of loading it: from the APK's assets as
 * the skill's `Classifier` does (A), from a file (B), from a file with XNNPACK's weight cache (C),
 * C with transient indirection buffers (D), from the assets on the GPU (E), D with the model closed
 * right after its task and created again (F), from a file on the GPU with D's CPU options (G), and
 * C with the XNNPACK flags set to the transient indirection bit alone, 0x20 (H).
 *
 * Each test reads the process at five points: P0 before the LiteRT environment exists, P1 after
 * `create`, the buffers and one warm-up run, P2 after the timed runs, P3 two seconds later with the
 * model still open (F: after `close`), and P4 after `close`. A point is `VmRSS`, `RssAnon`,
 * `RssFile` and `RssShmem` from `/proc/self/status`, the native heap in use, and the PSS and
 * private dirty pages of `Debug.MemoryInfo`. The result, with the load and run times and a hash of
 * the output for one fixed input, is written as JSON to `memprobe/` in the app's external files
 * directory and logged under the tag `MemoryProbe`.
 *
 * The models are not checked in: `-PprobeModels=<path>,<path>` packs `.tflite` files into the test
 * APK, and without them every test here is skipped. Run one test per process (one `am instrument`
 * per test), since what a process sets up once would otherwise count against the next test.
 *
 * Arguments (`-e <name> <value>`): `probeModel`, the asset (default `modnet.tflite`); `cache`,
 * `cold` to delete the weight cache file before the load or `warm` to keep it (default); `holdMs`,
 * a pause after P2 for `dumpsys meminfo` from the host (default 0); `iterations`, the timed runs
 * (default 10); `xnnFlags`, the XNNPACK flags of D, F and G in decimal (default 35 = 0x23:
 * transient indirection buffers, 0x20, on top of 0x3, which the value replaces rather than adds
 * to); `inputName`, to record that input's buffer requirements; `litertVersion`, for the result.
 */
@RunWith(AndroidJUnit4::class)
class MemoryProbeTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val testAssets: AssetManager = instrumentation.context.assets
    private val arguments = InstrumentationRegistry.getArguments()
    private val modelName = arguments.getString("probeModel") ?: "modnet.tflite"
    private val baseName = modelName.removeSuffix(".tflite")
    private val cache = arguments.getString("cache") ?: "warm"
    private val holdMs = arguments.getString("holdMs")?.toLong() ?: 0L
    private val iterations = arguments.getString("iterations")?.toInt() ?: 10
    private val xnnFlags = arguments.getString("xnnFlags")?.toInt() ?: DEFAULT_XNN_FLAGS
    private val inputName = arguments.getString("inputName")
    private val litertVersion = arguments.getString("litertVersion") ?: "unknown"

    /** A: `create(assets, name, Options(CPU), environment)`, step 2 of the skill. */
    @Test
    fun assetDefault() {
        probe(Variant("assetDefault", "A", fromFile = false, Accelerator.CPU))
    }

    /** B: copied once into `noBackupFilesDir`, then `create(path, Options(CPU), environment)`. */
    @Test
    fun filePath() {
        probe(Variant("filePath", "B", fromFile = true, Accelerator.CPU))
    }

    /** C: B with an XNNPACK weight cache file in `codeCacheDir`. */
    @Test
    fun filePathWeightCache() {
        probe(Variant("filePathWeightCache", "C", fromFile = true, Accelerator.CPU, cacheOf = "C"))
    }

    /** D: C with the transient indirection buffer flag added to the default flags. */
    @Test
    fun filePathWeightCacheTransientIndirection() {
        probe(
            Variant(
                "filePathWeightCacheTransientIndirection",
                "D",
                fromFile = true,
                Accelerator.CPU,
                cacheOf = "D",
                flags = xnnFlags,
            )
        )
    }

    /** E: A on the GPU. */
    @Test
    fun gpuAssetDefault() {
        probe(Variant("gpuAssetDefault", "E", fromFile = false, Accelerator.GPU))
    }

    /** F: D with its cache, closed right after the task, then created again and closed. */
    @Test
    fun closeAfterTask() {
        probe(
            Variant(
                "closeAfterTask",
                "F",
                fromFile = true,
                Accelerator.CPU,
                cacheOf = "D",
                flags = xnnFlags,
                closeAfterTask = true,
            )
        )
    }

    /** G: B on the GPU with D's CPU options. */
    @Test
    fun gpuFilePathCpuOptions() {
        probe(
            Variant(
                "gpuFilePathCpuOptions",
                "G",
                fromFile = true,
                Accelerator.GPU,
                cacheOf = "G",
                flags = xnnFlags,
            )
        )
    }

    /** H: C with the flags set to 0x20 alone, which also clears the default QS8 and QU8 bits. */
    @Test
    fun filePathWeightCacheTransientIndirectionOnly() {
        probe(
            Variant(
                "filePathWeightCacheTransientIndirectionOnly",
                "H",
                fromFile = true,
                Accelerator.CPU,
                cacheOf = "H",
                flags = TRANSIENT_INDIRECTION_ONLY,
            )
        )
    }

    /**
     * One way of loading the model. [cacheOf] names the variant whose cache file it uses; [flags]
     * are the XNNPACK flags it passes, none when null.
     */
    private class Variant(
        val test: String,
        val id: String,
        val fromFile: Boolean,
        val accelerator: Accelerator,
        val cacheOf: String? = null,
        val flags: Int? = null,
        val closeAfterTask: Boolean = false,
    )

    /** A model with its buffers, closed together, the buffers first. */
    private class Loaded(val model: CompiledModel) : AutoCloseable {
        val inputs = mutableListOf<TensorBuffer>()
        val outputs = mutableListOf<TensorBuffer>()

        init {
            try {
                inputs += model.createInputBuffers()
                outputs += model.createOutputBuffers()
            } catch (e: Exception) {
                close()
                throw e
            }
        }

        fun run(input: FloatArray): FloatArray {
            inputs[0].writeFloat(input)
            model.run(inputs, outputs)
            return outputs[0].readFloat()
        }

        override fun close() {
            (inputs + outputs).forEach { it.close() }
            model.close()
        }
    }

    private fun probe(variant: Variant) {
        val packed = testAssets.list("").orEmpty()
        assumeTrue(
            "$modelName is not in the test APK: build it with -PprobeModels=<path to $modelName>",
            modelName in packed,
        )
        val result = JSONObject()
        val points = JSONObject()
        val ms = JSONObject()
        result
            .put("model", modelName)
            .put("test", variant.test)
            .put("variant", variant.id)
            .put("accelerator", variant.accelerator.name)
            .put("cache", cacheLabel(variant))
            .put("xnnFlags", variant.flags ?: JSONObject.NULL)
            .put("iterations", iterations)
            .put("holdMs", holdMs)
            .put("litertVersion", litertVersion)
            .put("device", device())
            .put("pid", Process.myPid())
            .put("timestamp", SimpleDateFormat(ISO_TIME, Locale.US).format(Date()))
        points.put("P0", point("P0"))
        var failure: Throwable? = null
        try {
            measure(variant, result, points, ms)
        } catch (e: Throwable) {
            failure = e
            result.put("error", "${e.javaClass.simpleName}: ${e.message}")
        } finally {
            result.put("points", points).put("ms", ms)
            write(variant, result)
        }
        // A GPU that cannot take the model, or an input that does not read as float, is a result;
        // any other failure on the CPU fails the test.
        val recorded = failure is NotFloatInput || variant.accelerator == Accelerator.GPU
        if (failure != null && !recorded) {
            throw failure
        }
    }

    private class NotFloatInput(message: String) : Exception(message)

    private fun measure(variant: Variant, result: JSONObject, points: JSONObject, ms: JSONObject) {
        val staged = if (variant.fromFile) stage(result) else null
        val cacheFile =
            variant.cacheOf?.let { File(app.codeCacheDir, "$baseName.$it$CACHE_SUFFIX") }
        if (cacheFile != null) {
            val existed = cacheFile.exists()
            result.put("cacheFile", cacheFile.path).put("cacheExistedBefore", existed)
            result.put("cacheBytesBefore", if (existed) cacheFile.length() else 0L)
            if (cache == "cold" && existed) {
                check(cacheFile.delete()) { "could not delete ${cacheFile.path}" }
            }
        }
        val options =
            CompiledModel.Options(variant.accelerator).apply {
                if (cacheFile != null) {
                    cpuOptions =
                        CompiledModel.CpuOptions(
                            xnnPackFlags = variant.flags,
                            xnnPackWeightCachePath = cacheFile.path,
                        )
                }
            }
        val start = SystemClock.elapsedRealtimeNanos()
        val loaded = Loaded(create(staged, options).also { ms.put("create", msSince(start)) })
        try {
            val input = syntheticInput(inputElements(loaded, result))
            val warmUp = loaded.run(input)
            ms.put("createBuffersWarmUp", msSince(start))
            result.put("outputElements", warmUp.size).put("outputSha256WarmUp", sha256(warmUp))
            cacheFile?.let { result.put("cacheBytesAfterWarmUp", it.length()) }
            points.put("P1", point("P1"))
            val runs = JSONArray()
            var last = warmUp
            repeat(iterations) {
                val runStart = SystemClock.elapsedRealtimeNanos()
                last = loaded.run(input)
                runs.put(msSince(runStart))
            }
            ms.put("runs", runs)
            result.put("outputSha256Last", sha256(last))
            points.put("P2", point("P2"))
            if (holdMs > 0) {
                Thread.sleep(holdMs)
            }
            if (variant.closeAfterTask) {
                loaded.close()
                gc()
                points.put("P3", point("P3"))
                val again = SystemClock.elapsedRealtimeNanos()
                val model = create(staged, options)
                ms.put("recreate", msSince(again))
                Loaded(model).use { it.run(input) }
                ms.put("recreateBuffersWarmUpClose", msSince(again))
            } else {
                Thread.sleep(IDLE_MS)
                gc()
                points.put("P3", point("P3"))
                loaded.close()
            }
            gc()
            points.put("P4", point("P4"))
            cacheFile?.let { result.put("cacheBytesAtEnd", it.length()) }
        } finally {
            loaded.close()
        }
    }

    private fun create(staged: File?, options: CompiledModel.Options): CompiledModel =
        if (staged != null) {
            CompiledModel.create(staged.path, options, environment)
        } else {
            CompiledModel.create(testAssets, modelName, options, environment)
        }

    /**
     * The input's element count, from the buffer read back as floats. With `inputName`, also that
     * input's type and buffer size, which gives the count for a float input (4 bytes each).
     */
    private fun inputElements(loaded: Loaded, result: JSONObject): Int {
        val fromBuffer =
            try {
                loaded.inputs[0].readFloat().size
            } catch (e: Exception) {
                throw NotFloatInput("input 0 does not read as float: ${e.message}")
            }
        result.put("inputElementsFromBuffer", fromBuffer)
        val name = inputName ?: return fromBuffer
        result.put("inputName", name)
        try {
            result.put("inputType", loaded.model.getInputTensorType(name).elementType.name)
        } catch (e: Exception) {
            result.put("inputTypeError", "${e.javaClass.simpleName}: ${e.message}")
        }
        return try {
            val bytes = loaded.model.getInputBufferRequirements(name).bufferSize
            result.put("inputBufferBytes", bytes)
            if (bytes % 4 == 0 && bytes / 4 != fromBuffer) {
                result.put("inputElementsMismatch", "${bytes / 4} from the requirements")
            }
            if (bytes % 4 == 0) bytes / 4 else fromBuffer
        } catch (e: Exception) {
            result.put("inputBufferError", "${e.javaClass.simpleName}: ${e.message}")
            fromBuffer
        }
    }

    /** Copies the model once into `noBackupFilesDir`, again only when the size differs. */
    private fun stage(result: JSONObject): File {
        val staged = File(app.noBackupFilesDir, modelName)
        val size = assetBytes(result)
        val copy = !staged.isFile || staged.length() != size
        val start = SystemClock.elapsedRealtimeNanos()
        if (copy) {
            testAssets.open(modelName).use { input ->
                staged.outputStream().use { output -> input.copyTo(output, COPY_BUFFER_BYTES) }
            }
        }
        result.put(
            "staged",
            JSONObject()
                .put("path", staged.path)
                .put("bytes", staged.length())
                .put("copied", copy)
                .put("copyMs", if (copy) msSince(start) else 0.0),
        )
        return staged
    }

    /** The asset's size; a compressed asset has no file descriptor, so its bytes are counted. */
    private fun assetBytes(result: JSONObject): Long =
        try {
            val bytes = testAssets.openFd(modelName).use { it.length }
            result.put("assetCompressed", false)
            bytes
        } catch (e: FileNotFoundException) {
            result.put("assetCompressed", true)
            testAssets.open(modelName).use { input ->
                val buffer = ByteArray(COPY_BUFFER_BYTES)
                var total = 0L
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    total += read
                }
                total
            }
        }

    private fun point(name: String): JSONObject {
        val status = File("/proc/self/status").readLines()
        fun kb(key: String): Long =
            status
                .firstOrNull { it.startsWith("$key:") }
                ?.substringAfter(':')
                ?.trim()
                ?.substringBefore(' ')
                ?.toLongOrNull() ?: -1L
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        val point =
            JSONObject()
                .put("vmRssKb", kb("VmRSS"))
                .put("rssAnonKb", kb("RssAnon"))
                .put("rssFileKb", kb("RssFile"))
                .put("rssShmemKb", kb("RssShmem"))
                .put("nativeHeapBytes", Debug.getNativeHeapAllocatedSize())
                .put("totalPssKb", info.totalPss)
                .put("totalPrivateDirtyKb", info.totalPrivateDirty)
        Log.i(TAG, "POINT $name pid=${Process.myPid()} $point")
        return point
    }

    private fun write(variant: Variant, result: JSONObject) {
        val dir = File(app.getExternalFilesDir(null) ?: app.filesDir, "memprobe")
        dir.mkdirs()
        val stamp = SimpleDateFormat("yyyyMMdd'T'HHmmss", Locale.US).format(Date())
        val file = File(dir, "${baseName}_${variant.id}_${cacheLabel(variant)}_$stamp.json")
        result.put("file", file.path)
        file.writeText(result.toString(2))
        Log.i(TAG, "RESULT $result")
    }

    private fun cacheLabel(variant: Variant) = if (variant.cacheOf == null) "none" else cache

    private fun device(): JSONObject {
        val device =
            JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("model", Build.MODEL)
                .put("device", Build.DEVICE)
                .put("release", Build.VERSION.RELEASE)
                .put("sdk", Build.VERSION.SDK_INT)
                .put("fingerprint", Build.FINGERPRINT)
        if (Build.VERSION.SDK_INT >= 31) {
            device.put("soc", Build.SOC_MODEL)
        }
        return device
    }

    private fun gc() {
        repeat(2) {
            Runtime.getRuntime().gc()
            System.runFinalization()
        }
    }

    private fun msSince(startNanos: Long) = (SystemClock.elapsedRealtimeNanos() - startNanos) / 1e6

    /** The same values on every run and device, in -1..1. */
    private fun syntheticInput(size: Int) =
        FloatArray(size) { i -> (i.toLong() * 7919 % 2001).toFloat() / 1000f - 1f }

    private fun sha256(values: FloatArray): String {
        val bytes = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bytes.asFloatBuffer().put(values)
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes.array())
        return digest.joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val TAG = "MemoryProbe"
        const val ISO_TIME = "yyyy-MM-dd'T'HH:mm:ss.SSSZ"
        const val CACHE_SUFFIX = ".xnnpack_cache"
        const val IDLE_MS = 2000L
        const val COPY_BUFFER_BYTES = 1 shl 20

        /** 0x20 (transient indirection buffers) and 0x3 (QS8 and QU8, on by default). */
        const val DEFAULT_XNN_FLAGS = 0x23

        /** `TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER` alone. */
        const val TRANSIENT_INDIRECTION_ONLY = 0x20

        /** One environment for every model of the process, as in the skill. */
        val environment: Environment by lazy { Environment.create() }
    }
}
