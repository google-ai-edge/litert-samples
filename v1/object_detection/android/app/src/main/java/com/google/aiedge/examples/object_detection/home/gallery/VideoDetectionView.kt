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
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import com.google.android.exoplayer2.ui.StyledPlayerView
import com.google.aiedge.examples.object_detection.UiState
import com.google.aiedge.examples.object_detection.composables.ResultsOverlay
import com.google.aiedge.examples.object_detection.utils.getFittedBoxSize
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

// VideoDetectionView detects objects in a video periodically each couple of frames, and then plays
// the video while displaying results overlay on top of it and updating them as the video progresses

// It takes as an input the object detection options, a video uri, and function to set the inference
// time state

@Composable
fun VideoDetectionView(
    uiState: UiState,
    videoUri: Uri,
    onImageAnalyzed: (Bitmap) -> Unit,
) {
    // We first define some states

    // This state is used to indirectly control the video playback
    var isPlaying by remember {
        mutableStateOf(false)
    }

    // These two states hold the video dimensions, we don't know their values yet so we just set
    // them to 1x1 for now and updating them after loading the video
    var videoHeight by remember {
        mutableIntStateOf(1)
    }

    var videoWidth by remember {
        mutableIntStateOf(1)
    }

    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    // We use ExoPlayer to play our video from the uri with no sounds
    val exoPlayer = remember {
        ExoPlayer.Builder(context)
            .build()
            .also { exoPlayer ->
                val mediaItem = MediaItem.Builder()
                    .setUri(videoUri)
                    .build()
                exoPlayer.setMediaItem(mediaItem)
                exoPlayer.prepare()
                exoPlayer.volume = 0f
            }
    }

    // Now we run object detection on our video. For a better performance, we run it in background

    // On disposal of this composable, we release the exo player
    DisposableEffect(videoUri) {
        val retriever: MediaMetadataRetriever?
        val job: Job?
        retriever = MediaMetadataRetriever()
        isPlaying = true
        job = scope.launch(Dispatchers.IO) {
            // Load frames from the video.
            retriever.setDataSource(context, videoUri)
            val videoLengthMs =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    ?.toLong()

            // Note: We need to read width/height from frame instead of getting the width/height
            // of the video directly because MediaRetriever returns frames that are smaller than the
            // actual dimension of the video file.
            val firstFrame = retriever.getFrameAtTime(0)
            val width = firstFrame?.width
            val height = firstFrame?.height

            // If the video is invalid, returns a null
            if ((videoLengthMs == null) || (width == null) || (height == null)) return@launch

            // Next, we'll get one frame every frameInterval ms
            val numberOfFrameToRead = videoLengthMs.div(1000)
            for (i in 0..numberOfFrameToRead) {
                if (!isActive) return@launch
                val timestampMs = i * 1000
                val frame = retriever.getFrameAtTime(
                    timestampMs * 1000, // convert from ms to micro-s
                    MediaMetadataRetriever.OPTION_CLOSEST
                ) ?: return@launch
                onImageAnalyzed(frame)
            }
            retriever.release()
        }
        onDispose {
            retriever.release()
            job.cancel()
        }
    }


    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize(),
        contentAlignment = Alignment.TopCenter,
    ) {
        // When displaying the video, we want to scale it to fit in the available space, filling as
        // much space as it can with being cropped. While this behavior is easily achieved out of
        // the box with composables, we need the results overlay layer to have the exact same size
        // of the rendered video so that the results are drawn correctly on top of it. So we'll have
        // to calculate the size of the video after being scaled to fit in the available space
        // manually. To do that, we use the "getFittedBoxSize" function. Go to its implementation
        // for an explanation of how it works.

        val boxSize = getFittedBoxSize(
            containerSize = Size(
                width = this.maxWidth.value,
                height = this.maxHeight.value,
            ),
            boxSize = Size(
                width = videoWidth.toFloat(),
                height = videoHeight.toFloat()
            )
        )

        // Now that we have the exact UI size, we display the video and the results
        Box(
            modifier = Modifier
                .width(boxSize.width.dp)
                .height(boxSize.height.dp)
        ) {
            AndroidView(
                factory = {
                    StyledPlayerView(context).apply {
                        hideController()
                        useController = false
                        player = exoPlayer
                    }
                },
                // This block runs with every recomposition of the AndroidView, and we're using it
                // to play start playing the video as soon as "isPlaying" is set to true (which will
                // trigger a recomposition which in turn will trigger this bloc of code)
                update = {
                    if (isPlaying) {
                        videoHeight = exoPlayer.videoSize.height
                        videoWidth = exoPlayer.videoSize.width
                        exoPlayer.play()
                    }
                }
            )
            uiState.detectionResult?.let {
                ResultsOverlay(
                    modifier = Modifier
                        .width(boxSize.width.dp)
                        .height(boxSize.height.dp),
                    result = it,
                )
            }
        }
    }
    // While the object detection is in progress, we display a circular progress indicator
    if (!isPlaying) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
        ) {
            CircularProgressIndicator(
                modifier = Modifier.align(Alignment.Center)
            )
        }
    }
}
