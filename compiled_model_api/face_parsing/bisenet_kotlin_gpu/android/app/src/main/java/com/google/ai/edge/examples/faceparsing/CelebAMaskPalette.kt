/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.faceparsing
/** CelebAMask-HQ 19 face-parsing classes: names + overlay colors (ARGB). */
object CelebAMaskPalette {
    val NAMES = arrayOf(
        "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eyeglass",
        "l_ear", "r_ear", "earring", "nose", "mouth", "u_lip", "l_lip",
        "neck", "necklace", "cloth", "hair", "hat"
    )

    private val RGB = arrayOf(
        intArrayOf(0, 0, 0), intArrayOf(204, 153, 130), intArrayOf(0, 255, 0),
        intArrayOf(0, 204, 0), intArrayOf(0, 200, 255), intArrayOf(0, 130, 255),
        intArrayOf(255, 255, 0), intArrayOf(255, 150, 0), intArrayOf(255, 110, 0),
        intArrayOf(255, 220, 90), intArrayOf(230, 120, 200), intArrayOf(160, 0, 220),
        intArrayOf(255, 0, 120), intArrayOf(220, 0, 60), intArrayOf(180, 120, 90),
        intArrayOf(120, 200, 200), intArrayOf(90, 120, 220), intArrayOf(120, 70, 40),
        intArrayOf(60, 60, 160)
    )

    /** Background transparent; every other class an opaque overlay color. */
    val COLORS: IntArray = IntArray(RGB.size) { i ->
        val a = if (i == 0) 0 else 0xFF
        (a shl 24) or (RGB[i][0] shl 16) or (RGB[i][1] shl 8) or RGB[i][2]
    }
}
