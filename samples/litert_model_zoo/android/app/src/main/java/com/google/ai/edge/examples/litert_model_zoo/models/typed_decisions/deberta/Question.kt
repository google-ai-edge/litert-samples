// Vendored from https://huggingface.co/litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT/blob/7a276235b795e8ad3ae7ac6a9f237daa2098863a/android/sample/app/src/main/java/com/opendecision/Question.kt (Apache-2.0)
// formatted for this repository's 100-column and brace rules; no logic change
// one comment no longer names the source model
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.deberta

/** One typed question about the state, as the author's `decide()` takes it. */
data class Question(val kind: Kind, val instructions: String, val options: List<String>) {
  /** The three question kinds of the source model's API. */
  enum class Kind(val key: String) {
    /** One of 2 to 255 unordered options. */
    CHOICE("choice"),
    /** One of 2 to 10 ordered levels; the answer is the expected level index. */
    SCORE("score"),
    /** Yes or no; the answer is p(yes). Options are fixed to no, yes. */ NOUL("noul");

    companion object {
      fun fromKey(key: String): Kind =
        entries.firstOrNull { it.key == key }
          ?: throw IllegalArgumentException("Unknown question type \"$key\"")
    }
  }

  companion object {
    /** The fixed `noul` options of the author's schema: index 1 is "yes". */
    val NOUL_OPTIONS = listOf("no", "yes")

    fun choice(instructions: String, options: List<String>) =
      Question(Kind.CHOICE, instructions, options)

    fun score(instructions: String, options: List<String>) =
      Question(Kind.SCORE, instructions, options)

    fun noul(instructions: String) = Question(Kind.NOUL, instructions, NOUL_OPTIONS)

    /**
     * The author's question checks: `choice` 2 to 255 options, `score` 2 to 10, `noul` exactly the
     * fixed pair, non-empty instructions. Duplicate options are rejected as the author's wire
     * validator does.
     */
    fun validate(questions: List<Question>) {
      require(questions.isNotEmpty()) { "Add at least one question." }
      questions.forEachIndexed { index, q ->
        require(q.instructions.isNotBlank()) {
          "Question ${index + 1}: the instructions are empty."
        }
        when (q.kind) {
          Kind.CHOICE ->
            require(q.options.size in 2..255) {
              "Question ${index + 1}: choice takes 2 to 255 options."
            }
          Kind.SCORE ->
            require(q.options.size in 2..10) {
              "Question ${index + 1}: score takes 2 to 10 ordered levels."
            }
          Kind.NOUL ->
            require(q.options == NOUL_OPTIONS) { "Question ${index + 1}: noul has no options." }
        }
        require(q.options.all { it.isNotEmpty() }) { "Question ${index + 1}: an option is empty." }
        require(q.options.toSet().size == q.options.size) {
          "Question ${index + 1}: options repeat."
        }
      }
    }

    /**
     * Parses the app's question editor: one question per line, `choice: instructions | option,
     * option, …`, `score: instructions | level, level, …` or `noul: statement`. Blank lines are
     * skipped. Options cannot contain commas in this format.
     */
    fun parseLines(text: String): List<Question> =
      text.lines().filter { it.isNotBlank() }.mapIndexed { lineIndex, line ->
        val colon = line.indexOf(':')
        require(colon > 0) { "Line ${lineIndex + 1}: write \"choice: question | option, option\"." }
        val kind = Kind.fromKey(line.substring(0, colon).trim().lowercase())
        val rest = line.substring(colon + 1)
        val bar = rest.indexOf('|')
        val instructions = (if (bar >= 0) rest.substring(0, bar) else rest).trim()
        val options =
          if (bar >= 0) {
            rest.substring(bar + 1).split(',').map { it.trim() }.filter { it.isNotEmpty() }
          } else {
            emptyList()
          }
        when (kind) {
          Kind.NOUL -> {
            require(options.isEmpty()) { "Line ${lineIndex + 1}: noul takes no options." }
            noul(instructions)
          }
          Kind.CHOICE -> choice(instructions, options)
          Kind.SCORE -> score(instructions, options)
        }
      }
  }
}
