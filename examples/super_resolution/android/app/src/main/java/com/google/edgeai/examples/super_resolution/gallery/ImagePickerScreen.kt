package com.google.edgeai.examples.super_resolution.gallery

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.FloatingActionButton
import androidx.compose.material.Icon
import androidx.compose.material.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.edgeai.examples.super_resolution.ImageSuperResolutionHelper
import java.io.InputStream

@Composable
fun ImagePickerScreen(
    modifier: Modifier = Modifier,
    delegate: ImageSuperResolutionHelper.Delegate,
    onInferenceTimeCallback: (Int) -> Unit,
) {
    val context = LocalContext.current

    val viewModel: ImagePickerViewModel =
        viewModel(factory = ImagePickerViewModel.getFactory(context))
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }


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

    BoxWithConstraints(modifier = modifier.fillMaxSize()) {
        if (uiState.originalBitmap != null) {
            LazyVerticalGrid(
                modifier = Modifier.fillMaxSize(),
                columns = GridCells.Fixed(uiState.originalBitmap!!.height / maxWidth.value.toInt()),
                verticalArrangement = Arrangement.spacedBy(1.dp),
                horizontalArrangement = Arrangement.spacedBy(1.dp)
            ) {
                val gridBitmap = uiState.bitmapList
                items(gridBitmap.size) { index ->
                    Image(
                        modifier = Modifier.clickable {
                            selectedBitmap = gridBitmap[index]
                            viewModel.selectSubBitmap(index)
                        }, bitmap = gridBitmap[index].asImageBitmap(), contentDescription = null
                    )
                }
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

    if (selectedBitmap != null) {
        DialogWithImage(
            onDismiss = {
                selectedBitmap = null
                viewModel.resetSelectedBitmap()
            },
            onConfirmation = { viewModel.makeSharpen(selectedBitmap!!) },
            srcBitmap = selectedBitmap!!,
            dstBitmap = uiState.sharpenBitmap
        )
    }
}

@Composable
fun DialogWithImage(
    onDismiss: () -> Unit,
    onConfirmation: () -> Unit,
    srcBitmap: Bitmap,
    dstBitmap: Bitmap?,
) {
    Dialog(onDismissRequest = { onDismiss() }) {
        Card(
            modifier = Modifier
                .fillMaxSize()
                .padding(bottom = 80.dp),
            shape = RoundedCornerShape(16.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Image(
                    modifier = Modifier.size(200.dp),
                    bitmap = srcBitmap.asImageBitmap(),
                    contentDescription = null,
                    contentScale = ContentScale.FillWidth,
                )

                Spacer(modifier = Modifier.height(10.dp))

                if (dstBitmap != null) {
                    Image(
                        modifier = Modifier.size(200.dp),
                        bitmap = dstBitmap.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.FillWidth,
                    )
                }

                Spacer(modifier = Modifier.weight(1f))

                Row(
                    horizontalArrangement = Arrangement.Center,
                ) {
                    Button(
                        onClick = { onDismiss() },
                        modifier = Modifier.padding(8.dp),
                    ) {
                        Text("Close")
                    }
                    Button(
                        onClick = { onConfirmation() },
                        modifier = Modifier.padding(8.dp),
                    ) {
                        Text("Sharpen")
                    }
                }
            }

        }
    }
}
