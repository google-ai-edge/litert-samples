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

package com.google.ai.edge.examples.pidnet

/** Cityscapes 19 training classes: names + standard overlay colors (ARGB). */
object CityscapesPalette {
    val NAMES = arrayOf(
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle"
    )

    // Official Cityscapes colors (R, G, B).
    private val RGB = arrayOf(
        intArrayOf(128, 64, 128), intArrayOf(244, 35, 232), intArrayOf(70, 70, 70),
        intArrayOf(102, 102, 156), intArrayOf(190, 153, 153), intArrayOf(153, 153, 153),
        intArrayOf(250, 170, 30), intArrayOf(220, 220, 0), intArrayOf(107, 142, 35),
        intArrayOf(152, 251, 152), intArrayOf(70, 130, 180), intArrayOf(220, 20, 60),
        intArrayOf(255, 0, 0), intArrayOf(0, 0, 142), intArrayOf(0, 0, 70),
        intArrayOf(0, 60, 100), intArrayOf(0, 80, 100), intArrayOf(0, 0, 230),
        intArrayOf(119, 11, 32)
    )

    /** Opaque ARGB color per class (alpha applied by the overlay paint). */
    val COLORS: IntArray = IntArray(RGB.size) { i ->
        (0xFF shl 24) or (RGB[i][0] shl 16) or (RGB[i][1] shl 8) or RGB[i][2]
    }
}
