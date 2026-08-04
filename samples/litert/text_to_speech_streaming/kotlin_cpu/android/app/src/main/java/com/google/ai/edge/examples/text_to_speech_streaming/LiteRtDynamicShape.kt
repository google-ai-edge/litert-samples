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

package com.google.ai.edge.examples.text_to_speech_streaming

import com.google.ai.edge.litert.CompiledModel

/**
 * Temporary JNI bindings over the LiteRT C API's dynamic-shape entry points, which the Kotlin
 * `CompiledModel` API does not expose yet (see `cpp/dynamic_shape_jni.cc`).
 *
 * For a graph whose *output* shapes are statically inferable from the input shapes, the sequence
 * per inference is: [resizeInputTensor] for every input, [updateOutputLayouts] once (re-allocates
 * and propagates the new shapes to the outputs), then the standard `createInputBuffers` /
 * `createOutputBuffers` / `run`.
 *
 * When an output is a TFLite *dynamic* tensor — its shape only materializes during invoke, as with
 * this sample's vocoder waveform — pre-invoke buffer requirements are stale on the output side, so
 * [runDynamic] performs the whole inference in native code with caller-provided output shapes
 * instead.
 *
 * Delete this file (and the cpp/ directory) once the official Kotlin resize API ships.
 */
object LiteRtDynamicShape {

  init {
    System.loadLibrary("dynamic_shape_jni")
  }

  /**
   * Resizes one input tensor of [model] — `LiteRtCompiledModelResizeInputTensor`. Requires the
   * resized dimensions to be dynamic (`-1`) in the model signature.
   *
   * @throws RuntimeException if LiteRT rejects the resize.
   */
  fun resizeInputTensor(
    model: CompiledModel,
    signatureIndex: Int,
    inputIndex: Int,
    dims: IntArray,
  ) {
    nativeResizeInputTensor(model, signatureIndex, inputIndex, dims)
  }

  /**
   * Re-allocates the model after input resizes and returns the propagated output shapes —
   * `LiteRtGetCompiledModelOutputTensorLayouts(update_allocation=true)`. Call between the last
   * [resizeInputTensor] and `createOutputBuffers`. For TFLite-dynamic outputs the returned shape is
   * still the placeholder (their shape only exists during invoke) — use [runDynamic] then.
   *
   * @throws RuntimeException if LiteRT fails to re-allocate.
   */
  fun updateOutputLayouts(
    model: CompiledModel,
    signatureIndex: Int,
    outputCount: Int,
  ): Array<IntArray> = nativeUpdateOutputLayouts(model, signatureIndex, outputCount)

  /**
   * One complete float32 inference with per-call shapes: strict-resizes every input, runs with
   * exact-sized host-memory buffers on both sides, and returns the outputs. [outputShapes] must be
   * the true final shapes — for this sample's vocoder they are known analytically from the frame
   * count.
   *
   * @throws RuntimeException if any LiteRT call fails.
   */
  fun runDynamic(
    model: CompiledModel,
    signatureIndex: Int,
    inputs: Array<FloatArray>,
    inputShapes: Array<IntArray>,
    outputShapes: Array<IntArray>,
  ): Array<FloatArray> = nativeRunDynamic(model, signatureIndex, inputs, inputShapes, outputShapes)

  private external fun nativeResizeInputTensor(
    model: CompiledModel,
    signatureIndex: Int,
    inputIndex: Int,
    dims: IntArray,
  )

  private external fun nativeUpdateOutputLayouts(
    model: CompiledModel,
    signatureIndex: Int,
    outputCount: Int,
  ): Array<IntArray>

  private external fun nativeRunDynamic(
    model: CompiledModel,
    signatureIndex: Int,
    inputs: Array<FloatArray>,
    inputShapes: Array<IntArray>,
    outputShapes: Array<IntArray>,
  ): Array<FloatArray>
}
