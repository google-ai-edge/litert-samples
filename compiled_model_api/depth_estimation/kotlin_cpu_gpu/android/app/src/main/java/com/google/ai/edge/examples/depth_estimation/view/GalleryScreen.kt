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

package com.google.ai.edge.examples.depth_estimation.view

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import coil3.asDrawable
import coil3.compose.AsyncImage
import coil3.imageLoader
import com.google.ai.edge.examples.depth_estimation.UiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

@Composable
fun GalleryScreen(
    uiState: UiState,
    modifier: Modifier = Modifier,
    onImageAnalyzed: (Bitmap) -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val uri = uiState.mediaUri
    val mediaType = getMediaType(context, uri)

    DisposableEffect(uri) {
        var retriever: MediaMetadataRetriever? = null
        var job: Job? = null
        if (mediaType == MediaType.VIDEO) {
            retriever = MediaMetadataRetriever()
            job =
                scope.launch(Dispatchers.IO) {
                    // Load frames from the video.
                    retriever.setDataSource(context, uri)
                    val videoLengthMs =
                        retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()

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
                        val timestampUs = i * 1000000
                        val frame =
                            retriever.getFrameAtTime(timestampUs, MediaMetadataRetriever.OPTION_CLOSEST)
                                ?: return@launch
                        onImageAnalyzed(frame)
                    }
                    retriever.release()
                }
        }
        onDispose {
            retriever?.release()
            job?.cancel()
        }
    }

    Box(modifier = modifier.background(Color.Black)) {
        when (mediaType) {
            MediaType.IMAGE -> {
                AsyncImage(
                    modifier = Modifier.fillMaxSize(),
                    model = uri,
                    contentDescription = null,
                    // Use the imageLoader extension property on Context
                    imageLoader = LocalContext.current.imageLoader,
                    onSuccess = { state ->
                        // Coil 3 hands back AsyncImagePainter.State.Success; the
                        // decoded image lives on its `result`, not the state itself.
                        val coilImage = state.result.image
                        val drawable = coilImage.asDrawable(context.resources)
                        val bitmap = (drawable as? android.graphics.drawable.BitmapDrawable)?.bitmap

                        bitmap?.let {
                            onImageAnalyzed(it)
                        }
                    },
                    contentScale = ContentScale.Fit,
                )
            }

            MediaType.VIDEO -> {
                AndroidView(
                    modifier = Modifier.fillMaxSize(),
                    factory = { ctx ->
                        ScalableVideoView(ctx).apply {
                            setDisplayMode(ScalableVideoView.DisplayMode.ORIGINAL)
                            setVideoURI(uri)
                            setOnPreparedListener { it.start() }
                        }
                    },
                    update = {
                        it.setVideoURI(uri)
                        it.setOnPreparedListener { player ->
                            // start() is a no-op when the player is already playing.
                            player.start()
                        }
                    },
                )
            }

            MediaType.UNKNOWN -> {
                // Do Nothing
            }
        }

        if (uiState.overlay != null) {
            DepthOverlay(
                modifier = Modifier.fillMaxSize(),
                overlay = uiState.overlay,
                lensFacing = uiState.lensFacing,
            )
        }
    }
}

enum class MediaType {
    IMAGE,
    VIDEO,
    UNKNOWN,
}

fun getMediaType(context: Context, uri: Uri): MediaType {
    val mimeType = context.contentResolver.getType(uri)
    return if (mimeType?.startsWith("image/") == true) {
        MediaType.IMAGE
    } else if (mimeType?.startsWith("video/") == true) {
        MediaType.VIDEO
    } else {
        MediaType.UNKNOWN
    }
}
