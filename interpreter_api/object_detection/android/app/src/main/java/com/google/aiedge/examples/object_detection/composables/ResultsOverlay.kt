/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.aiedge.examples.object_detection.composables

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.google.aiedge.examples.object_detection.objectdetector.ObjectDetectorHelper
import com.google.aiedge.examples.object_detection.ui.Turquoise

// This composable is used to display the results of the object detection

// Beside results, it also needs to know the dimensions of the media (video, image, etc.)
// on which the object detection was performed so that it can draw the results properly.

// This information is needed because each result bounds are calculated based on the
// dimensions of the original media that the object detection was performed on. But for
// us to draw the bounds correctly, we need to draw the bounds based on the dimensions of
// the UI space that the media is being displayed in.

// An important note is that this composable should have the exact same UI dimensions of the
// media being displayed, and it should be placed exactly on the top of the displayed media.
// For example, if an image is being displayed in a Box composable, the overlay should be placed
// on top of the image and it should fill the Box composable.
// This is a must because it scales the result bounds according to the provided frame dimensions
// as well as the max available UI width and height


@Composable
fun ResultsOverlay(
    modifier: Modifier = Modifier,
    result: ObjectDetectorHelper.DetectionResult,
) {
    BoxWithConstraints(
        modifier
            .fillMaxSize()
    ) {
        val detections = result.detections
        for (detection in detections) {
            val ratioBox = detection.boundingBox
            val boxWidth = ratioBox.width() * maxWidth.value
            val boxHeight = ratioBox.height() * maxHeight.value
            val boxLeftOffset = ratioBox.left * maxWidth.value
            val boxTopOffset = ratioBox.top * maxHeight.value
            Box(
                Modifier
                    .fillMaxSize()
                    .offset(
                        boxLeftOffset.dp,
                        boxTopOffset.dp,
                    )
                    .width(boxWidth.dp)
                    .height(boxHeight.dp)
            ) {
                Box(
                    modifier = Modifier
                        .border(3.dp, Turquoise)
                        .width(boxWidth.dp)
                        .height(boxHeight.dp)
                )
                Box(modifier = Modifier.padding(3.dp)) {
                    Text(
                        text = "${
                            detection.label
                        } ${String.format("%.1f", detection.score)}",
                        modifier = Modifier
                            .background(Color.Black)
                            .padding(5.dp, 0.dp),
                        color = Color.White,
                    )
                }
            }
        }
    }
}
