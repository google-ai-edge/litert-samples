package com.google.edgeai.examples.super_resolution.gallery

import android.graphics.BitmapFactory
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.FloatingActionButton
import androidx.compose.material.Icon
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.edgeai.examples.super_resolution.ImageSuperResolutionHelper
import java.io.InputStream
import kotlin.math.roundToInt

@Composable
fun ImagePickerScreen(
    modifier: Modifier = Modifier,
    delegate: ImageSuperResolutionHelper.Delegate,
    onInferenceTimeCallback: (Int) -> Unit,
) {
    val context = LocalContext.current
    val density = LocalDensity.current

    val viewModel: ImagePickerViewModel =
        viewModel(factory = ImagePickerViewModel.getFactory(context))
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val scrollState = rememberScrollState()

    LaunchedEffect(key1 = uiState.inferenceTime) {
        onInferenceTimeCallback(uiState.inferenceTime)
    }

    LaunchedEffect(key1 = delegate) {
        viewModel.setDelegate(delegate)
    }

    // Register ActivityResult handler
    val imagePicker =
        rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            if (uri != null) {
                val contentResolver = context.contentResolver
                val inputStream: InputStream? = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                viewModel.selectImage(bitmap)
            }
        }

    BoxWithConstraints(
        modifier = modifier
            .fillMaxSize(),
    ) {
        if (uiState.originalBitmap != null) {
            Image(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(scrollState)
                    .pointerInput(Unit) {
                        detectTapGestures { offset ->
                            with(density) {
                                viewModel.selectOffset(
                                    offset,
                                    Size(maxWidth.toPx(), maxHeight.toPx())
                                )
                            }
                        }
                    }
                    .detectDrag(
                        onDrag = {
                            with(density) {
                                viewModel.selectOffset(
                                    it,
                                    Size(maxWidth.toPx(), maxHeight.toPx())
                                )
                            }
                        },
                    ),
                bitmap = uiState.originalBitmap!!.asImageBitmap(),
                contentDescription = null,
                contentScale = ContentScale.FillWidth
            )

            val selectPoint = uiState.selectPoint
            if (selectPoint.offset != null) {
                Box(
                    modifier = Modifier
                        .size(
                            (selectPoint.boxSize / density.density).dp,
                            (selectPoint.boxSize / density.density).dp
                        )
                        .offset {
                            IntOffset(
                                selectPoint.offset.x.roundToInt(),
                                selectPoint.offset.y.roundToInt(),
                            )
                        }
                        .border(border = BorderStroke(width = 3.dp, color = Color.Green))
                )
            }

            if (uiState.sharpenBitmap != null) {
                Image(
                    modifier = Modifier
                        .size(120.dp, 120.dp)
                        .align(Alignment.TopEnd),
                    bitmap = uiState.sharpenBitmap!!.asImageBitmap(),
                    contentDescription = null,
                )
            }
        }

        FloatingActionButton(modifier = Modifier
            .align(Alignment.BottomEnd)
            .padding(bottom = 80.dp, end = 16.dp),
            shape = CircleShape,
            onClick = {
                val request =
                    PickVisualMediaRequest(mediaType = ActivityResultContracts.PickVisualMedia.ImageOnly)
                imagePicker.launch(request)
            }) {
            Icon(Icons.Filled.Add, contentDescription = null)
        }
    }
}

/**
 * Detects drag gestures on a composable element.
 *
 * @param onDrag Callback invoked during a drag gesture.
 * @return A modifier that applies drag gesture detection to the composable.
 */
fun Modifier.detectDrag(
    onDrag: (Offset) -> Unit
): Modifier = composed {
    val interactionSource = remember {
        MutableInteractionSource()
    }
    this.pointerInput(interactionSource) {
        detectDragGestures(
            onDragStart = { },
            onDragEnd = { },
        ) { change, _ ->
            change.consume()
            onDrag(change.position)
        }
    }
}
