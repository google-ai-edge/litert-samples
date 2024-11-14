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

package com.google.aiedge.examples.digit_classifier

import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.PaintingStyle
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.aiedge.examples.digit_classifier.ui.ApplicationTheme


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

        setContent {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()

            ApplicationTheme {
                Scaffold(
                    topBar = {
                        Header()
                    },
                ) { paddingValue ->
                    Column(
                        modifier = Modifier.padding(paddingValue),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        // Board composable for drawing and interaction
                        Board(
                            modifier = Modifier.fillMaxWidth(),
                            drawOffsets = uiState.drawOffsets,
                            onDragStart = {
                                viewModel.draw(it)
                            },
                            onDrag = {
                                viewModel.draw(it)
                            },
                            onDragEnd = {},
                            onDraw = {
                                if (uiState.drawOffsets.isNotEmpty()) {
                                    viewModel.classify(it)
                                }
                            },
                        )

                        Spacer(modifier = Modifier.height(30.dp))
                        Text(stringResource(id = R.string.result, uiState.digit))
                        Spacer(modifier = Modifier.height(5.dp))
                        Text(stringResource(id = R.string.confidence, uiState.score))
                        Spacer(Modifier.weight(1f))
                        Button(onClick = {
                            viewModel.cleanBoard()
                        }) {
                            Text(stringResource(id = R.string.clear))
                        }
                    }

                }
            }
        }
    }

    @Composable
    fun Board(
        modifier: Modifier = Modifier,
        drawOffsets: List<DrawOffset>,
        onDrag: (DrawOffset) -> Unit,
        onDragStart: (DrawOffset) -> Unit,
        onDragEnd: () -> Unit,
        onDraw: (Bitmap) -> Unit,
    ) {
        val screenWidth = LocalConfiguration.current.screenWidthDp.dp
        val localDensity = LocalDensity.current

        val path by remember { mutableStateOf(Path()) }  // Use mutablePathOf for updates

        val bitmap = remember {
            val width = with(localDensity) { screenWidth.toPx() }.toInt()
            Bitmap.createBitmap(width, width, Bitmap.Config.ARGB_8888)
        }

        val bitmapCanvas = remember {
            androidx.compose.ui.graphics.Canvas(bitmap.asImageBitmap())
        }

        LaunchedEffect(key1 = drawOffsets) {
            path.reset()  // Reset path directly instead of creating a new one
            bitmap.eraseColor(android.graphics.Color.TRANSPARENT)
        }

        val paint = remember {
            Paint().apply {
                color = Color.White
                style = PaintingStyle.Stroke
                strokeWidth = 70f
                strokeCap = StrokeCap.Round
            }
        }

        BoxWithConstraints(
            modifier = modifier
                .fillMaxWidth()
                .height(screenWidth)
                .background(MaterialTheme.colorScheme.primary)
                .detectDrag(
                    onDragStart = {
                        onDragStart(Start(it.x, it.y))
                    },
                    onDragEnd = {
                        onDragEnd()
                    },
                    onDrag = {
                        onDrag(Point(it.x, it.y))
                    },
                )
        ) {
            Canvas(
                modifier = Modifier
                    .width(this.maxWidth)
                    .height(this.maxWidth)
            ) {
                drawOffsets.forEach { offset ->
                    when (offset) {
                        is Start -> path.moveTo(offset.x, offset.y)
                        is Point -> path.lineTo(offset.x, offset.y)
                    }
                }
                bitmapCanvas.drawPath(paint = paint, path = path)
                this.drawPath(
                    path = path,
                    color = Color.White,
                    style = Stroke(width = 70f, cap = StrokeCap.Round)
                )

                onDraw(bitmap)
            }
        }
    }


    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun Header() {
        TopAppBar(
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.secondary,
            ),
            title = {
                Image(
                    modifier = Modifier.size(120.dp),
                    alignment = Alignment.CenterStart,
                    painter = painterResource(id = R.drawable.logo),
                    contentDescription = null,
                )
            },
        )
    }
}


/**
 * Detects drag gestures on a composable element.
 *
 * @param onDragStart Callback invoked when a drag gesture starts.
 * @param onDragEnd Callback invoked when a drag gesture ends.
 * @param onDrag Callback invoked during a drag gesture.
 * @return A modifier that applies drag gesture detection to the composable.
 */
fun Modifier.detectDrag(
    onDragStart: (Offset) -> Unit, onDragEnd: () -> Unit, onDrag: (Offset) -> Unit
): Modifier = composed {
    val interactionSource = remember {
        MutableInteractionSource()
    }
    this.pointerInput(interactionSource) {
        detectDragGestures(
            onDragStart = onDragStart,
            onDragEnd = onDragEnd,
        ) { change, _ ->
            change.consume()
            onDrag(change.position)
        }
    }
}