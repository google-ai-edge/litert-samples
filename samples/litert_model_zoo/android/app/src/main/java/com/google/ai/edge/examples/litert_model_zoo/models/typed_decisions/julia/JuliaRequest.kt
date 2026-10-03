// Vendored from https://huggingface.co/litert-community/Julia-1-LiteRT/blob/8f36857c56e891c023060586759c6cdc8baf6b3e/android/app/src/main/java/com/julia1/JuliaRequest.kt (Apache-2.0)
// SPDX-License-Identifier: Apache-2.0
// Port of SupersonicLabs/Julia-1 julia/data.py sequence() with strict encoding and the named
// questions of julia/typed.py, as julia_litert.py in litert-community/Julia-1-LiteRT does.
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.julia

/** The request does not fit the window without truncation; strict encoding rejects it. */
class EncodingException(message: String) : IllegalArgumentException(message)

/** One named question: `choice`, `score` or `noul`, with the criteria of the author's API. */
data class JuliaQuestion(val type: String, val instructions: String, val criteria: Any?) {
  /** Caller IDs returned with the probabilities: mapping keys, zero-based indices, false/true. */
  val keys: List<String>

  /** Option texts scored at the markers: the criteria descriptions themselves, no prefixes. */
  val options: List<String>

  init {
    when (type) {
      "choice" -> {
        val mapping =
          criteria as? Map<*, *> ?: error("Choice criteria must map IDs to descriptions")
        require(mapping.keys.all { it is String && it.isNotEmpty() }) {
          "Choice criteria must map nonempty IDs to descriptions"
        }
        keys = mapping.keys.map { it as String }
        options = mapping.values.map { description(it) }
      }
      "score" -> {
        val rubric = criteria as? List<*> ?: error("Score requires an ordered rubric")
        keys = rubric.indices.map { it.toString() }
        options = rubric.map { description(it) }
      }
      "noul" -> {
        keys = listOf("false", "true")
        options =
          if (criteria == null) {
            keys
          } else {
            val mapping = criteria as? Map<*, *> ?: error("Noul criteria must map false and true")
            require(mapping.keys.map { it.toString() }.toSet() == keys.toSet()) {
              "Noul criteria must map false and true to descriptions"
            }
            keys.map { description(mapping[it]) }
          }
      }
      else -> error("Unsupported question type: $type")
    }
    require(options.size in 2..20) { "options must contain 2-20 nonempty descriptions" }
    require(type != "noul" || options.size == 2) { "noul options must be ordered [false, true]" }
  }

  /** Graph channel of `qtype_onehot`: choice = 0, score = 1, noul = 2. */
  val qtype: Int
    get() =
      when (type) {
        "choice" -> 0
        "score" -> 1
        else -> 2
      }

  companion object {
    /** Reads the author's question object: `type`, `instructions` and `criteria`. */
    fun fromMap(question: Map<String, Any?>): JuliaQuestion =
      JuliaQuestion(
        question["type"] as? String ?: error("Question requires a string type"),
        question["instructions"] as? String ?: error("Question requires string instructions"),
        question["criteria"],
      )

    /** The author's validate_row accepts only nonempty strings as rendered descriptions. */
    private fun description(value: Any?): String =
      (value as? String)?.takeIf { it.isNotEmpty() }
        ?: error("options must contain 2-20 nonempty descriptions")
  }
}

/** Unpadded token IDs and one marker position per option for one graph invocation. */
class JuliaSequence(val ids: IntArray, val markers: IntArray, val question: JuliaQuestion) {
  /** Graph channel of `qtype_onehot`. */
  val qtype: Int
    get() = question.qtype
}

/**
 * Builds one request row exactly as the author's strict encoding does: nothing is truncated, and a
 * request that does not fit raises [EncodingException]. Under strict encoding the ids do not depend
 * on the window or on the head budget when nothing overflows.
 */
class JuliaSequenceBuilder(
  private val tokenizer: JuliaTokenizer,
  val window: Int = 512,
  /** Question plus options budget; any value below window - 4 keeps the author's strict rules. */
  val headLength: Int = minOf(512, window - 5),
) {
  init {
    require(window > 0 && headLength in 1 until window - 4) { "Invalid window or head budget" }
  }

  /** Text state passes through; a map or list state is serialized like json.dumps(). */
  fun build(state: Any?, question: JuliaQuestion): JuliaSequence {
    val stateText =
      when (state) {
        is String -> state
        is Map<*, *>,
        is List<*> -> JuliaJson.stringify(state)
        else -> throw IllegalArgumentException("state must be text or JSON")
      }
    require(
      (listOf(stateText, question.instructions) + question.options).none {
        it.contains(MASK_LITERAL)
      }
    ) {
      "Reserved model marker in request"
    }
    val head = tokenizer.encode("${question.type} question: ${question.instructions}")
    val optionIds = question.options.map { tokenizer.encode(" $it") }
    if (optionIds.any { it.size > MAX_OPTION_TOKENS }) {
      throw EncodingException("Option exceeds the $MAX_OPTION_TOKENS-token model contract")
    }
    val budget = headLength - optionIds.sumOf { it.size + 1 }
    if (budget < 16) {
      // The author's squeeze would cut options here; strict encoding rejects instead.
      val perOption = maxOf(4, Math.floorDiv(headLength - 16, optionIds.size))
      if (optionIds.any { it.size + 1 > perOption }) {
        throw EncodingException("Options exceed the head budget")
      }
    }
    if (head.size > budget) {
      throw EncodingException("Question and options exceed the head budget")
    }
    val ids = ArrayList<Int>(window)
    ids.add(CLS)
    head.forEach { ids.add(it) }
    ids.add(SEP)
    val markers = IntArray(optionIds.size)
    optionIds.forEachIndexed { index, option ->
      markers[index] = ids.size
      ids.add(MASK)
      option.forEach { ids.add(it) }
    }
    ids.add(SEP)
    val stateIds = tokenizer.encode(stateText)
    val needed = ids.size + stateIds.size + 1
    if (needed > window) {
      throw EncodingException("Request needs $needed tokens; the window is $window")
    }
    stateIds.forEach { ids.add(it) }
    ids.add(SEP)
    return JuliaSequence(ids.toIntArray(), markers, question)
  }

  companion object {
    /** `<pad>` of the checkpoint tokenizer; padding rows gather this real embedding. */
    const val PAD = 0
    /** `<eos>` separates the question head, the options and the state. */
    const val SEP = 1
    /** `<bos>` starts every sequence. */
    const val CLS = 2
    /** `<mask>` marks the position whose logit scores the option that follows it. */
    const val MASK = 4
    /** Tokens allowed per option after its marker. */
    const val MAX_OPTION_TOKENS = 48
    private const val MASK_LITERAL = "<mask>"
  }
}
