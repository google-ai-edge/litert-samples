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

package com.google.ai.edge.examples.image_cutout

import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/**
 * The files `install_to_device.sh` puts in the app's `files/models/`, with the size and SHA-256 the
 * app checks before it builds a graph. Change a file here and in the install script together.
 */
object ModelFiles {
  class Spec(val name: String, val bytes: Long, val sha256: String, val what: String)

  val VISION_ENCODER =
    Spec(
      "wevisone_vision_encoder_512_fp16.tflite",
      611_814_624L,
      "47d4089213cfdb6da006263ac92cfee92e8e2ff36c4e59027e4fe16dce2e837a",
      "vision encoder",
    )
  val VISION_ADAPTER =
    Spec(
      "wevisone_vision_adapter_512_fp16.tflite",
      201_460_224L,
      "d0c8dc73b2a5a358a3bb06438d643cadbc454245f011d89853d184d9d646c530",
      "vision adapter",
    )
  val DECODER_FP16 =
    Spec(
      "wevisone_decoder_b_fp16.tflite",
      3_458_517_312L,
      "56dc97e27115ab8be102c8e8eb543895735a0eb2d15367fd7a44a736b3ee9864",
      "language model (fp16 weights)",
    )
  val DECODER_INT8 =
    Spec(
      "wevisone_decoder_b_int8.tflite",
      1_785_462_176L,
      "f93bf881dccd203a1e55d8fa43cf51798739a5aa5fe441ad7cb3b53808579a6f",
      "language model (int8 weights)",
    )
  val MASK_DECODER =
    Spec(
      "wevisone_mask_decoder_gs16.tflite",
      21_878_000L,
      "ac0765dc75ce81708564421bfe930a7714b589bae43edac2ddc810c57fa5f7e1",
      "mask decoder",
    )
  val EMBEDDINGS =
    Spec(
      "wevisone_embed_tokens_fp16.bin",
      625_377_280L,
      "5ff0fcb29afce21c5efc2c717fe723e4329443373513ba53b7c33d18508050fe",
      "token embedding table",
    )
  val TOKENIZER =
    Spec(
      "tokenizer.json",
      11_607_393L,
      "3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e",
      "tokenizer",
    )

  fun common(): List<Spec> =
    listOf(TOKENIZER, EMBEDDINGS, VISION_ENCODER, VISION_ADAPTER, MASK_DECODER)

  fun dir(filesDir: File): File = File(filesDir, "models")

  /** A file problem the app reports instead of building a graph. */
  class Problem(val spec: Spec, val message: String)

  /**
   * Missing files and wrong sizes (cheap: before every run); with [hash] also the SHA-256 (once per
   * install: the result is remembered by size and modification time in [verified]).
   */
  fun check(
    dir: File,
    specs: List<Spec>,
    hash: Boolean,
    verified: MutableMap<String, String>,
  ): List<Problem> {
    val out = ArrayList<Problem>()
    for (spec in specs) {
      val f = File(dir, spec.name)
      if (!f.isFile) {
        out.add(Problem(spec, "Missing model file: ${spec.name} (${spec.what})."))
        continue
      }
      if (f.length() != spec.bytes) {
        out.add(Problem(spec, "${spec.name} is ${f.length()} bytes, expected ${spec.bytes}."))
        continue
      }
      if (!hash) continue
      val key = "${f.length()}:${f.lastModified()}"
      if (verified[spec.name] == key) continue
      val digest = sha256(f)
      if (digest != spec.sha256) {
        out.add(
          Problem(
            spec,
            "${spec.name} has SHA-256 ${digest.take(12)}..., expected ${spec.sha256.take(12)}...",
          )
        )
      } else {
        verified[spec.name] = key
      }
    }
    return out
  }

  fun sha256(file: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    FileInputStream(file).use { input ->
      val buf = ByteArray(1 shl 20)
      while (true) {
        val n = input.read(buf)
        if (n < 0) break
        md.update(buf, 0, n)
      }
    }
    return md.digest().joinToString("") { "%02x".format(it) }
  }

  fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}
