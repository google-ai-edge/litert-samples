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

package com.google.ai.edge.examples.ocr_verification

import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/**
 * The files of litert-community/MiniCPM-OCR-Decision-2B-LiteRT the app reads from its private
 * `files/models/` (copied there by `install_to_device.sh`), with the size and SHA-256 of the
 * repository's `contract.json`. The tokenizer, the tables and at least one tower and one decision
 * file are required; any other single graph file of the list may be added, and requests then use
 * the smallest file that holds them.
 */
object ModelFiles {
  class Spec(val name: String, val bytes: Long, val sha256: String, val what: String)

  val TOKENIZER = Spec("tokenizer.json", 19_989_325L, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", "tokenizer")
  val TABLES =
    Spec(
      "tables.safetensors",
      1_021_845_928L,
      "8fa765826c3676521fa3cc6af61f1cc2c783b129c9564051a0abc28d51678986",
      "embedding tables",
    )

  /** Every single graph file of the repository (the multi-signature files are for desktops). */
  val GRAPHS =
    listOf(
      Spec("decide_L256_f16sfc.tflite", 2_750_394_480L, "526daaa7071c1ffb924844856259c9987cef8e6df88cbc2766f78c03286f2999", "decision graph"),
      Spec("decide_L384_f16sfc.tflite", 2_750_868_256L, "c93ddd7239325ca92aa84f8fa681e610f07709228ae792afb20a16e12794e832", "decision graph"),
      Spec("decide_L512_f16sfc.tflite", 2_751_472_944L, "83a715086ee914c12771eb986556ea6d8f246794aeb4ebb2949c0456df767a91", "decision graph"),
      Spec("decide_L768_f16sfc.tflite", 2_753_075_504L, "a35df5ccaff271bc6b7b100cab04df3e81db6043739fb9d9d27cdafadb7a0163", "decision graph"),
      Spec("decide_L1024_f16sfc.tflite", 2_755_202_352L, "170842c13d4684fb8220eb92fd22b829e0a593a96171a457abe0e8930e59eb11", "decision graph"),
      Spec("decide_L1536_f16sfc.tflite", 2_761_028_912L, "ceeabf92b4c6bbce2087eb8dfb0aee4c64871b5c2d66eb12285b6bcab52c667a", "decision graph"),
      Spec("decide_L2048_f16sfc.tflite", 2_768_952_624L, "0f9b8f0384a6e158d7a24062f25f18fe088e5fb3401cf8de5cad9f90bf44d2ea", "decision graph"),
      Spec("decide_L3072_f16sfc.tflite", 2_791_091_504L, "088772270e2fb447e2cc292464b30e32f3983c3b34c70c2570f2b54626b9a419", "decision graph"),
      Spec("decide_L4096_qb1024_f16sfc.tflite", 2_758_845_216L, "c7c1b82bff8f7701623495d119028f0eb3a4f5286e7109cd9cb74febfcac804f", "decision graph"),
      Spec("tower_N400_fp16fc.tflite", 658_758_848L, "4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244", "picture tower"),
      Spec("tower_N576_fp16fc.tflite", 658_758_848L, "9742c0c2bd791f501e65eb752f86e18be75b42c7e5f3efd3e0af5f7155f49dcd", "picture tower"),
      Spec("tower_N784_fp16fc.tflite", 658_758_848L, "4d6cdc7130641db90a9036b8ec18187703de2218e631ca95db1a43ce080e6642", "picture tower"),
      Spec("tower_N1024_fp16fc.tflite", 658_758_848L, "1857499d81652664bd26585087f59b8eb09253e78cf1a0e43bcbb48078871424", "picture tower"),
      Spec("tower_N1296_fp16fc.tflite", 658_758_848L, "56a6a8d71ef2c67fc73303a4080af802a675961c05031d19bca4407407223b66", "picture tower"),
      Spec("tower_N1600_fp16fc.tflite", 658_758_848L, "347b77b7a3b034f90b9ac2c8fad2bd632555bfb2e9804367fbbe0aa9f89b3c37", "picture tower"),
      Spec("tower_N1936_fp16fc.tflite", 658_758_848L, "6bf103d1e4e2efeb9c42b9f420f690b30949ab2929790c2647c678df0ec8eaad", "picture tower"),
      Spec("tower_N2304_fp16fc.tflite", 658_758_848L, "e6d27ae15aacef61c638506e6a5cb030acd91d1a23238cd7ac2a4528c771675e", "picture tower"),
      Spec("tower_N3136_fp16fc.tflite", 658_758_848L, "0fe5b485bfb7c6e8b9d42a089ce0f159c096a0559a45c6c7c97ae3241339382c", "picture tower"),
      Spec("tower_N4096_qb1024_fp16fc.tflite", 658_857_872L, "15477014ac55979e77e43b3b4fbf5fe1ebfc0059038ec56274358aedcac76d6e", "picture tower"),
    )

  fun dir(filesDir: File): File = File(filesDir, "models")

  /** A file problem the app reports instead of loading a graph. */
  class Problem(val name: String, val message: String)

  /**
   * Missing files and wrong sizes (cheap: before every load); with [hash] also the SHA-256, once
   * per file version (remembered by size and modification time in [verified]).
   */
  fun check(dir: File, hash: Boolean, verified: MutableMap<String, String>): List<Problem> {
    val out = ArrayList<Problem>()
    val present = GRAPHS.filter { File(dir, it.name).isFile }
    if (present.none { it.name.startsWith("tower_") }) {
      out.add(Problem("tower", "No picture tower file (tower_N<patches>_fp16fc.tflite) in files/models."))
    }
    if (present.none { it.name.startsWith("decide_") }) {
      out.add(Problem("decide", "No decision file (decide_L<tokens>_f16sfc.tflite) in files/models."))
    }
    for (spec in listOf(TOKENIZER, TABLES) + present) {
      val f = File(dir, spec.name)
      if (!f.isFile) {
        out.add(Problem(spec.name, "Missing model file: ${spec.name} (${spec.what})."))
        continue
      }
      if (f.length() != spec.bytes) {
        out.add(Problem(spec.name, "${spec.name} is ${f.length()} bytes, expected ${spec.bytes}."))
        continue
      }
      if (!hash) continue
      val key = "${f.length()}:${f.lastModified()}"
      if (verified[spec.name] == key) continue
      val digest = sha256(f)
      if (digest != spec.sha256) {
        out.add(
          Problem(spec.name, "${spec.name} has SHA-256 ${digest.take(12)}..., expected ${spec.sha256.take(12)}...")
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
