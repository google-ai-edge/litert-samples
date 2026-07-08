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

package com.google.ai.edge.examples.rwkv7

import android.util.Half
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * RWKV-7 "World" 0.1B autoregressive text generation with the full per-token
 * forward pass on the LiteRT CompiledModel GPU delegate.
 *
 * RWKV is an RNN: one token per step, no KV cache growth. The whole recurrent
 * state lives host-side and is fed back every step (Mali corrupts stateful GPU
 * graphs, so nothing persists on the accelerator):
 *
 *   inputs : x_emb[1,768], att_shift[12,768], ffn_shift[12,768], wkv[144,64,64]
 *   outputs: logits[1,65536] and the three updated states, same shapes.
 *
 * The token embedding row is looked up host-side from a memory-mapped fp16
 * table (GATHER is not GPU-compatible); the first LayerNorm is inside the graph,
 * so the raw embedding row is the model input.
 */
class Rwkv7Generator(modelPath: String, embTablePath: String) : AutoCloseable {

    companion object {
        const val VOCAB_SIZE = 65536
        private const val N_LAYER = 12
        private const val N_EMBD = 768
        private const val HEAD_DIM = 64
        private const val N_HEAD = N_EMBD / HEAD_DIM
        private const val SHIFT_SIZE = N_LAYER * N_EMBD
        private const val WKV_SIZE = N_LAYER * N_HEAD * HEAD_DIM * HEAD_DIM
        private const val BYTES_PER_EMB_ROW = N_EMBD * 2

        /** Token 0 is the RWKV World end-of-text token (not present in the vocab file). */
        const val END_OF_TEXT = 0
    }

    /** Per-step timing, measured around run + output readback. */
    data class StepStats(val stepMs: Float)

    private val model =
        CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()

    private val embRows: ByteBuffer = RandomAccessFile(File(embTablePath), "r").use {
        it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length())
            .order(ByteOrder.LITTLE_ENDIAN)
    }

    private val embRow = FloatArray(N_EMBD)
    private var attShift = FloatArray(SHIFT_SIZE)
    private var ffnShift = FloatArray(SHIFT_SIZE)
    private var wkvState = FloatArray(WKV_SIZE)
    private var logits = FloatArray(VOCAB_SIZE)

    /** Clears all recurrent state; call before starting a new sequence. */
    fun reset() {
        attShift.fill(0f)
        ffnShift.fill(0f)
        wkvState.fill(0f)
    }

    /**
     * Runs one autoregressive step for [token] and returns the logits over the
     * next token. States are recycled host-side into the next call.
     */
    fun step(token: Int): FloatArray {
        lookupEmbedding(token)
        inputs[0].writeFloat(embRow)
        inputs[1].writeFloat(attShift)
        inputs[2].writeFloat(ffnShift)
        inputs[3].writeFloat(wkvState)
        model.run(inputs, outputs)
        logits = outputs[0].readFloat()
        attShift = outputs[1].readFloat()
        ffnShift = outputs[2].readFloat()
        wkvState = outputs[3].readFloat()
        return logits
    }

    /** Greedy decode: index of the highest logit. */
    fun argmax(logits: FloatArray): Int {
        var best = 0
        var bestValue = logits[0]
        for (i in 1 until logits.size) {
            if (logits[i] > bestValue) {
                bestValue = logits[i]
                best = i
            }
        }
        return best
    }

    /**
     * Feeds [promptIds] (prefill), then greedily generates up to [maxTokens].
     * Invokes [onToken] with each new token id and timing; return false from the
     * callback to stop early. Returns the generated ids.
     */
    fun generate(
        promptIds: IntArray,
        maxTokens: Int,
        onToken: (tokenId: Int, stats: StepStats) -> Boolean,
    ): IntArray {
        reset()
        var lastLogits = FloatArray(VOCAB_SIZE)
        for (id in promptIds) {
            lastLogits = step(id)
        }
        val generated = ArrayList<Int>(maxTokens)
        for (i in 0 until maxTokens) {
            val next = argmax(lastLogits)
            if (next == END_OF_TEXT) {
                break
            }
            generated.add(next)
            val start = System.nanoTime()
            lastLogits = step(next)
            val stepMs = (System.nanoTime() - start) / 1e6f
            if (!onToken(next, StepStats(stepMs))) {
                break
            }
        }
        return generated.toIntArray()
    }

    /** Reads one fp16 embedding row into [embRow] as float32. */
    private fun lookupEmbedding(token: Int) {
        var offset = token * BYTES_PER_EMB_ROW
        for (i in 0 until N_EMBD) {
            embRow[i] = Half.toFloat(embRows.getShort(offset))
            offset += 2
        }
    }

    override fun close() {
        model.close()
    }
}
