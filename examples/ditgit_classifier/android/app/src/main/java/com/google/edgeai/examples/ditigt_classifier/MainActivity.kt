package com.google.edgeai.examples.ditigt_classifier

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
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
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
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }


        setContent {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            Scaffold(
                topBar = {
                    Header()
                },
            ) { paddingValue ->
                Column(
                    modifier = Modifier.padding(paddingValue),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
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
                            viewModel.classify(it)
                        },
                    )

                    Spacer(modifier = Modifier.height(30.dp))
                    Text("Prediction result: ${uiState.digit}")
                    Spacer(modifier = Modifier.height(5.dp))
                    Text("Confidence: ${uiState.score}")
                    Spacer(Modifier.weight(1f))
                    Button(onClick = {
                        viewModel.cleanBoard()
                    }) {
                        Text("Clear")
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

        val paint = remember {
            Paint().apply {
                color = Color.White
                style = PaintingStyle.Stroke
                strokeWidth = 50f
                strokeCap = StrokeCap.Round
            }
        }

        BoxWithConstraints(
            modifier = modifier
                .fillMaxWidth()
                .height(screenWidth)
                .background(Color.Black)
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
            if (drawOffsets.isNotEmpty()) {
                val path = Path()

                val bitmap = Bitmap.createBitmap(
                    with(localDensity) { maxWidth.toPx() }.toInt(),
                    with(localDensity) { maxWidth.toPx() }.toInt(),
                    Bitmap.Config.ARGB_8888
                )

                Canvas(
                    modifier = Modifier
                        .width(this.maxWidth)
                        .height(this.maxWidth)
                ) {
                    val bitmapCanvas = androidx.compose.ui.graphics.Canvas(bitmap.asImageBitmap())

                    if (drawOffsets.isNotEmpty()) {
                        drawOffsets.forEach {
                            if (it is Start) {
                                path.moveTo(it.x, it.y)
                            }
                            if (it is Point) {
                                path.lineTo(it.x, it.y)
                            }
                        }
                        bitmapCanvas.drawPath(
                            paint = paint, path = path
                        )
                        drawIntoCanvas {
                            it.nativeCanvas.drawBitmap(bitmap, 0f, 0f, null)
                        }
                        onDraw(bitmap)
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun Header(modifier: Modifier = Modifier) {
        TopAppBar(
            modifier = modifier,
            colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.LightGray),
            title = {
                Image(
                    modifier = Modifier.fillMaxSize(),
                    painter = painterResource(id = R.drawable.tfl_logo),
                    contentDescription = null,
                )
            },
        )
    }
}


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
