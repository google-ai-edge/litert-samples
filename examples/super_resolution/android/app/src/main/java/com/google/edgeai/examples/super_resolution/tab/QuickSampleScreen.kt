package com.google.edgeai.examples.super_resolution.tab

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.google.edgeai.examples.super_resolution.UiState
import java.io.InputStream

@Composable
fun QuickSampleScreen(
    uiState: UiState,
    modifier: Modifier = Modifier,
    onMakeSharpen: (Bitmap) -> Unit,
) {
    val context = LocalContext.current
    var selectUri by remember {
        mutableStateOf<String?>(null)
    }
    Column {
        Row(modifier) {
            uiState.sampleUriList.forEach {
                AsyncImage(
                    modifier = Modifier
                        .size(70.dp)
                        .clickable {
                            selectUri = it
                        },
                    model = "file:///android_asset/$it",
                    contentDescription = null
                )
            }
        }

        if (selectUri != null) {
            Button(onClick = {
                val assetManager = context.assets
                val inputStream: InputStream = assetManager.open(selectUri.toString())
                val bitmap = BitmapFactory.decodeStream(inputStream)
                onMakeSharpen(bitmap)
            }) {
                Text(text = "UPSAMPLE")
            }

            AsyncImage(
                modifier = Modifier.size(150.dp),
                model = "file:///android_asset/$selectUri",
                contentDescription = "",
            )
        }

        if (uiState.sharpenBitmap != null) {
            Image(
                modifier = Modifier.size(150.dp),
                bitmap = uiState.sharpenBitmap.asImageBitmap(), contentDescription = null
            )
        }
    }
}