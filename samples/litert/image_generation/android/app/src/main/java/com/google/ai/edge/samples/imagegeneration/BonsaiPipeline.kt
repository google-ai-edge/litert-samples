// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Bonsai Image 4B on-device pipeline for Android (port of the iOS app ../ios/Sources/
// BonsaiPipeline.swift). Three fixed-shape .tflite graphs over the LiteRT
// Interpreter API on CPU/XNNPACK, loaded sequentially and closed between
// stages so peak memory stays ~DiT-sized rather than the ~4 GiB sum — the
// difference between running and being LMK-killed on 8 GB devices.

package com.google.ai.edge.samples.imagegeneration

import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class BonsaiPipeline(private val modelsDir: File, meta: JSONObject) {

    class MissingModels(val files: List<String>) : Exception("missing: $files")
    class Cancelled : Exception()

    private val ditFile = meta.getJSONObject("files").getString("dit")
    private val textencFile = meta.getJSONObject("files").getString("textenc")
    private val vaeFile = meta.getJSONObject("files").getString("vae")
    private val bnScale = meta.getJSONArray("latent_bn_scale").let { a ->
        FloatArray(a.length()) { a.getDouble(it).toFloat() }
    }
    private val bnShift = meta.getJSONArray("latent_bn_shift").let { a ->
        FloatArray(a.length()) { a.getDouble(it).toFloat() }
    }

    @Volatile var cancelled = false

    companion object {
        val THREADS = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)

        /** The published file name, or its `_fixed` sibling (the zero-scale-
         *  patched DiT used during device verification). */
        fun resolveModel(name: String, dir: File): File? =
            listOf(name, name.replace(".tflite", "_fixed.tflite"))
                .map { File(dir, it) }.firstOrNull { it.exists() }

        fun missingFiles(modelsDir: File, meta: JSONObject): List<String> {
            val f = meta.getJSONObject("files")
            return listOf("dit", "textenc", "vae").map { f.getString(it) }
                .filter { resolveModel(it, modelsDir) == null }
        }
    }

    class Result(
        val rgb: ByteArray,            // 512*512*3
        val seconds: Double,
        val stepSeconds: List<Double>,
    )

    // MARK: single fixed-shape graph, CPU/XNNPACK

    private class Graph(file: File, threads: Int) : AutoCloseable {
        val interp: Interpreter
        val loadSeconds: Double
        private val argOrder: IntArray   // argOrder[k] = graph input index of arg k

        init {
            val t = System.nanoTime()
            // File-path constructor mmaps the flatbuffer — mandatory for the
            // 2.11 GiB DiT; a heap ByteBuffer copy would double the footprint.
            interp = Interpreter(
                file,
                Interpreter.Options().apply {
                    numThreads = threads
                    setUseXNNPACK(true)   // required for blockwise-int4 correctness + speed
                }
            )
            interp.allocateTensors()
            loadSeconds = (System.nanoTime() - t) / 1e9
            // input order by serving_default_args_<n>, NEVER by shape/index
            val n = interp.inputTensorCount
            fun argpos(i: Int): Int {
                val name = interp.getInputTensor(i).name()
                val at = name.lastIndexOf("args_")
                if (at < 0) return i
                return name.substring(at + 5).takeWhile { it.isDigit() }.toIntOrNull() ?: i
            }
            argOrder = (0 until n).sortedBy { argpos(it) }.toIntArray()
        }

        /** inputs in ARGUMENT order; returns output tensor 0 as floats. */
        fun run(inputs: List<ByteBuffer>, outCount: Int): FloatArray {
            require(inputs.size == argOrder.size) {
                "inputCount ${inputs.size} != ${argOrder.size}"
            }
            val byGraphIndex = arrayOfNulls<Any>(inputs.size)
            for ((argIdx, graphIdx) in argOrder.withIndex()) {
                val t = interp.getInputTensor(graphIdx)
                val buf = inputs[argIdx]
                buf.rewind()
                require(t.numBytes() == buf.capacity()) {
                    "byteSize input $graphIdx: graph ${t.numBytes()} != host ${buf.capacity()}"
                }
                byGraphIndex[graphIdx] = buf
            }
            val out = ByteBuffer.allocateDirect(outCount * 4).order(ByteOrder.nativeOrder())
            interp.runForMultipleInputsOutputs(byGraphIndex, mapOf(0 to out))
            out.rewind()
            val floats = FloatArray(outCount)
            out.asFloatBuffer().get(floats)
            return floats
        }

        override fun close() = interp.close()
    }

    private fun modelFile(name: String): File =
        resolveModel(name, modelsDir) ?: throw MissingModels(listOf(name))

    private fun checkCancel() {
        if (cancelled) throw Cancelled()
    }

    private fun fbuf(a: FloatArray): ByteBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder())
            .apply { asFloatBuffer().put(a); rewind() }

    private fun ibuf(a: IntArray): ByteBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder())
            .apply { asIntBuffer().put(a); rewind() }

    // MARK: generation

    fun generate(
        tokenizer: QwenTokenizer,
        prompt: String,
        seed: Long,
        steps: Int,
        status: (String) -> Unit,
        progress: (String, Double) -> Unit,
    ): Result {
        val t00 = System.nanoTime()
        cancelled = false

        // ---- stage 1: tokenize + text encoder ------------------------------
        progress("Encoding prompt…", 0.02)
        val enc = tokenizer.encodePrompt(prompt)
        status("prompt: ${enc.promptTokenCount} tokens")
        var embeds: FloatArray
        Graph(modelFile(textencFile), THREADS).use { te ->
            checkCancel()
            val t = System.nanoTime()
            embeds = te.run(
                listOf(ibuf(enc.ids), ibuf(enc.mask)),
                BonsaiMath.SEQ * 7680
            )
            status("text encoder %.1fs (load %.1fs)"
                .format((System.nanoTime() - t) / 1e9, te.loadSeconds))
        }
        checkCancel()

        // ---- stage 2: DiT Euler loop ----------------------------------------
        progress("Loading DiT (2.1 GiB)…", 0.10)
        val sigmas = BonsaiMath.sigmas(steps)
        val imgIds = fbuf(BonsaiMath.imgIds())
        val txtIds = fbuf(BonsaiMath.txtIds())
        val embedsBuf = fbuf(embeds)
        var lat = BonsaiMath.noise(seed)
        val stepSeconds = ArrayList<Double>(steps)
        Graph(modelFile(ditFile), THREADS).use { dit ->
            status("DiT loaded %.1fs".format(dit.loadSeconds))
            for (k in 0 until steps) {
                checkCancel()
                progress("Step ${k + 1} of $steps…", 0.16 + 0.72 * k / steps)
                val t = System.nanoTime()
                val v = dit.run(
                    listOf(fbuf(lat), embedsBuf, fbuf(floatArrayOf(sigmas[k])), imgIds, txtIds),
                    BonsaiMath.TOKENS * BonsaiMath.PACKED_CHANNELS
                )
                val ds = sigmas[k + 1] - sigmas[k]
                for (i in lat.indices) lat[i] += ds * v[i]
                stepSeconds.add((System.nanoTime() - t) / 1e9)
                status("step ${k + 1}/$steps  sigma %.3f  %.1fs"
                    .format(sigmas[k], stepSeconds[k]))
            }
        }
        checkCancel()

        // ---- stage 3: unpatchify + VAE decode -------------------------------
        progress("Decoding image…", 0.90)
        val z = BonsaiMath.unpatchify(lat, bnScale, bnShift)
        val rgb = ByteArray(512 * 512 * 3)
        Graph(modelFile(vaeFile), THREADS).use { vae ->
            val t = System.nanoTime()
            val y = vae.run(listOf(fbuf(z)), 3 * 512 * 512)
            for (c in 0 until 3) for (p in 0 until 512 * 512) {
                val v = ((y[c * 262144 + p] / 2f + 0.5f) * 255f).coerceIn(0f, 255f)
                // round half-up == Swift .rounded() for non-negative values
                rgb[p * 3 + c] = Math.round(v).toByte()
            }
            status("VAE decode %.1fs".format((System.nanoTime() - t) / 1e9))
        }

        val total = (System.nanoTime() - t00) / 1e9
        progress("Done", 1.0)
        return Result(rgb, total, stepSeconds)
    }
}
