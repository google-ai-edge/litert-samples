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

package com.google.ai.edge.examples.zero_shot_classification

import java.io.File

/**
 * Only temperature fields are used; calibration provenance remains available in the source JSON.
 */
class LayaCalibration(
  private val temperatures: DoubleArray = doubleArrayOf(1.0, 1.0, 1.0),
  private val byOptions: Map<String, Double> = emptyMap(),
) {
  init {
    require(temperatures.size == 3) { "Expected choice/score/noul temperatures" }
  }

  /** Uses the option-count bucket when present, then the question-type temperature. */
  fun temperature(qtype: String, k: Int): Double =
    byOptions[bucket(qtype, k)]
      ?: temperatures[
        when (qtype) {
          "choice" -> 0
          "score" -> 1
          "noul" -> 2
          else -> error("Unknown question type: $qtype")
        }]

  companion object {
    /** Leaves logits uncalibrated for the captured T=1 reference gate. */
    fun identity(): LayaCalibration = LayaCalibration()

    /** Loads calibration from an installed JSON file. */
    fun load(file: File): LayaCalibration = fromMap(LayaJson.asObject(LayaJson.parse(file)))

    /** Reads only temperature fields; unrelated provenance fields remain unused. */
    fun fromMap(config: Map<String, Any?>): LayaCalibration {
      val defaults =
        config["temperature"]?.let { value ->
          LayaJson.asArray(value).map { (it as Number).toDouble() }.toDoubleArray()
        } ?: doubleArrayOf(1.0, 1.0, 1.0)
      val buckets =
        config["temperature_by_options"]?.let { value ->
          LayaJson.asObject(value).mapValues { (_, number) -> (number as Number).toDouble() }
        } ?: emptyMap()
      return LayaCalibration(defaults, buckets)
    }

    /** Matches the host contract's two-, five-, and ten-option bucket boundaries. */
    fun bucket(qtype: String, k: Int): String =
      qtype +
        ":" +
        when {
          k <= 2 -> "2"
          k <= 5 -> "3-5"
          k <= 10 -> "6-10"
          else -> "11+"
        }
  }
}
