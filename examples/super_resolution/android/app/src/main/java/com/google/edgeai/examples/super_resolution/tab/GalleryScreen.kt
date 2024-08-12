package com.google.edgeai.examples.super_resolution.tab

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.Card
import androidx.compose.material.Text
import androidx.compose.material.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.google.edgeai.examples.super_resolution.UiState

@Composable
fun ImagePickerScreen(
    uiState: UiState, onMakeSharpen: (Bitmap) -> Unit,
) {
    var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    val gridBitmap = uiState.bitmapList
    LazyVerticalGrid(
        columns = GridCells.Fixed(maxWidth.value.toInt() / 200),
        verticalArrangement = Arrangement.spacedBy(1.dp),
        horizontalArrangement = Arrangement.spacedBy(1.dp)
    ) {
        items(gridBitmap.size) { index ->
            Image(
                modifier = Modifier.clickable {
                    selectedBitmap = gridBitmap[index]
                },
                bitmap = gridBitmap[index].asImageBitmap(),
                contentDescription = null
            )
        }
    }

    if (selectedBitmap != null) {
        DialogWithImage(
            onDismissRequest = { selectedBitmap = null },
            onConfirmation = { onMakeSharpen(selectedBitmap!!) },
            srcBitmap = selectedBitmap!!,
            dstBitmap = uiState.sharpenBitmap
        )
    }
}

@Composable
fun DialogWithImage(
    onDismissRequest: () -> Unit,
    onConfirmation: () -> Unit,
    srcBitmap: Bitmap,
    dstBitmap: Bitmap?,
) {
    Dialog(onDismissRequest = { onDismissRequest() }) {
        Card(
            modifier = Modifier
                .fillMaxSize()
                .height(375.dp)
                .padding(16.dp),
            shape = RoundedCornerShape(16.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Image(
                    modifier = Modifier
                        .size(200.dp),
                    bitmap = srcBitmap.asImageBitmap(),
                    contentDescription = null,
                    contentScale = ContentScale.FillWidth,
                )

                if (dstBitmap != null) {
                    Image(
                        modifier = Modifier
                            .size(200.dp),
                        bitmap = dstBitmap.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.FillWidth,
                    )
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center,
                ) {
                    TextButton(
                        onClick = { onDismissRequest() },
                        modifier = Modifier.padding(8.dp),
                    ) {
                        Text("Dismiss")
                    }
                    TextButton(
                        onClick = { onConfirmation() },
                        modifier = Modifier.padding(8.dp),
                    ) {
                        Text("Confirm")
                    }
                }
            }

        }
    }
}
