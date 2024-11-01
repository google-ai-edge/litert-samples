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

package com.google.aiedge.examples.object_detection.home.gallery

import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.absoluteOffset
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.aiedge.examples.object_detection.UiState
import com.google.aiedge.examples.object_detection.ui.teal

// Here we have the gallery view which is displayed in Home screen

// It's used to run object detection on images and videos from the phone's storage

// It takes as input the object detection options, and a function to update the inference time state
@RequiresApi(Build.VERSION_CODES.P)
@Composable
fun GalleryScreen(
    uiState: UiState,
    modifier: Modifier = Modifier,
    onMediaPicked: () -> Unit,
    onImageAnalyzed: (Bitmap) -> Unit
) {
    // We need a state to hold the Uri of the currently chosen media (image or video) if any
    var selectedMediaUri by rememberSaveable {
        mutableStateOf<Uri?>(null)
    }

    // We use this launcher later to launch a new activity to select a media file from
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
        // On selecting an image or a video, we update the selected media uri
        onResult = { uri ->
            onMediaPicked()
            selectedMediaUri = uri
        },
    )

    Box(modifier = modifier.fillMaxSize()) {
        // We first want to figure out the type of the selected media, which is one of
        // the following three cases: Image, Video, Unknown.

        val selectedMediaType = selectedMediaUri?.let {
            val mimeType = LocalContext.current.contentResolver.getType(it)
            if (mimeType == null) {
                "Unknown"
            } else if (mimeType.startsWith("image")) {
                "Image"
            } else if (mimeType.startsWith("video")) {
                "Video"
            } else {
                "Unknown"
            }
        }

        // Now that we know the selected media type, we display the appropriate composable
        when (selectedMediaType) {
            "Image" -> ImageDetectionView(
                uiState = uiState,
                imageUri = selectedMediaUri!!,
                onImageAnalyzed = {
                    onImageAnalyzed(it)
                }
            )

            "Video" -> VideoDetectionView(
                uiState = uiState,
                videoUri = selectedMediaUri!!,
                onImageAnalyzed = {
                    onImageAnalyzed(it)
                }
            )

            "Unknown" -> Text(
                text = "Unknown media type",
                modifier = Modifier.align(
                    Alignment.Center
                ),
            )
            // In this case, the selected media uri is null, i.e, nothing is selected
            else -> Text(
                text = "Click + to add an image or a video to begin running the object detection.",
                modifier = Modifier
                    .align(
                        Alignment.Center
                    )
                    .padding(20.dp),
                textAlign = TextAlign.Center,
                color = Color.Gray,
                fontSize = 13.sp
            )
        }

        // The floating action button here is used to launch an activity to select a media file from
        FloatingActionButton(
            onClick = {
                selectedMediaUri = null
                galleryLauncher.launch(arrayOf("image/*", "video/*"))
            },
            containerColor = teal,
            contentColor = Color.White,
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .absoluteOffset((-20).dp, (-75).dp)
        ) {
            Icon(
                Icons.Filled.Add,
                contentDescription = null,
            )
        }
    }
}
