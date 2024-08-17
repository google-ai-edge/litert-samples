package com.google.edgeai.examples.super_resolution.quicksample

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.google.edgeai.examples.super_resolution.ImageSuperResolutionHelper
import java.io.InputStream

@Composable
fun QuickSampleScreen(
    modifier: Modifier = Modifier,
    delegate: ImageSuperResolutionHelper.Delegate,
    onInferenceTimeCallback: (Int) -> Unit,
) {
    val context = LocalContext.current
    val viewModel: QuickSampleViewModel =
        viewModel(factory = QuickSampleViewModel.getFactory(context))

    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val selectBitmap = uiState.selectBitmap

    LaunchedEffect(key1 = uiState.inferenceTime) {
        onInferenceTimeCallback(uiState.inferenceTime)
    }

    LaunchedEffect(key1 = delegate) {
        viewModel.setDelegate(delegate)
    }

    Column(modifier.padding(8.dp)) {
        Text(text = "Choose a low resolution image below:")
        Spacer(modifier = Modifier.height(5.dp))

        Row(modifier) {
            uiState.sampleUriList.forEach {
                AsyncImage(
                    modifier = Modifier
                        .size(70.dp)
                        .clickable {
                            val assetManager = context.assets
                            val inputStream: InputStream =
                                assetManager.open(it)
                            val bitmap = BitmapFactory.decodeStream(inputStream)
                            viewModel.selectImage(bitmap)
                        },
                    model = "file:///android_asset/$it",
                    contentDescription = null
                )
            }
        }

        Text(
            text = "Click UPSAMPLE button below will we use TFLite to generate a corresponding " +
                    "high resolution image based n your chosen image"
        )
        Spacer(modifier = Modifier.height(5.dp))

        Row {
            if (selectBitmap != null) {
                AsyncImage(
                    modifier = Modifier.size(150.dp),
                    model = selectBitmap,
                    contentDescription = "",
                )
            }

            Spacer(modifier = Modifier.width(10.dp))

            if (uiState.sharpenBitmap != null) {
                Image(
                    modifier = Modifier.size(150.dp),
                    bitmap = uiState.sharpenBitmap!!.asImageBitmap(), contentDescription = null
                )
            }
        }
        if (selectBitmap != null) {
            Button(onClick = {
                viewModel.makeSharpen()
            }) {
                Text(text = "UPSAMPLE")
            }
        }
    }
}