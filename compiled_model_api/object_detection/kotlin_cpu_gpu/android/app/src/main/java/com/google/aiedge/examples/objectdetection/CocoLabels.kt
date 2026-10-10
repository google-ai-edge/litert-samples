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

package com.google.aiedge.examples.objectdetection

import android.graphics.Color

/** The 80 contiguous COCO classes used by YOLOX (index 0-79), plus a per-class color. */
object CocoLabels {
    private val NAMES = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    )

    fun name(classId: Int): String = NAMES.getOrElse(classId) { "obj$classId" }

    /**
     * COCO 91-way category ids (1-90 with gaps, id 0 = background) for each contiguous NAMES index.
     * DETR-family heads (RF-DETR) classify in this id space.
     */
    private val COCO91_IDS = intArrayOf(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    )

    /** Contiguous 0-79 index for a COCO 91-space category id, or -1 (background / unused ids). */
    fun index80(cocoId: Int): Int = COCO91_IDS.indexOf(cocoId)

    /** Deterministic per-class color (ARGB), evenly spread around the hue wheel. */
    fun color(classId: Int): Int {
        val hue = (classId * 47 % 360).toFloat()
        return Color.HSVToColor(floatArrayOf(hue, 0.85f, 1.0f))
    }
}
