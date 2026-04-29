package com.example.mobilenetlitert

data class Classification(
  val label: String,
  val score: Float,
  val index: Int,
)

data class ClassificationResult(
  val predictions: List<Classification>,
  val backend: String,
  val preprocessMs: Long,
  val inferenceMs: Long,
)
