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

package com.google.ai.edge.examples.pose_estimation.view

import android.content.pm.PackageManager
import android.util.Log
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.ai.edge.examples.pose_estimation.UiState
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val TAG = "CameraScreen"

@Composable
fun CameraScreen(
  uiState: UiState,
  modifier: Modifier = Modifier,
  onImageAnalyzed: (ImageProxy) -> Unit,
) {
  val context = LocalContext.current

  val launcher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
      isGranted: Boolean ->
      if (!isGranted) {
        Toast.makeText(context, "Camera permission is denied", Toast.LENGTH_SHORT).show()
      }
    }

  LaunchedEffect(key1 = uiState.errorMessage) {
    if (
      ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) !=
        PackageManager.PERMISSION_GRANTED
    ) {
      launcher.launch(android.Manifest.permission.CAMERA)
    }
  }

  // Live preview with the detected skeleton drawn on top.
  Box(modifier = modifier.fillMaxWidth().background(Color.Black)) {
    CameraPreview(
      modifier = Modifier.fillMaxSize(),
      onImageAnalyzed = { imageProxy -> onImageAnalyzed(imageProxy) },
      lensFacing = uiState.lensFacing,
    )
    PoseOverlay(
      keypoints = uiState.keypoints,
      sourceWidth = uiState.sourceWidth,
      sourceHeight = uiState.sourceHeight,
      lensFacing = uiState.lensFacing,
      modifier = Modifier.fillMaxSize(),
    )
  }
}

@Composable
fun CameraPreview(
  modifier: Modifier = Modifier,
  onImageAnalyzed: (ImageProxy) -> Unit,
  lensFacing: Int,
) {
  val context = LocalContext.current
  val lifecycleOwner = LocalContext.current as LifecycleOwner
  val cameraProviderFuture by remember {
    mutableStateOf(ProcessCameraProvider.getInstance(context))
  }
  val cameraExecutor: ExecutorService = remember { Executors.newSingleThreadExecutor() }

  val previewView = remember {
    PreviewView(context).apply {
      layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
      // FIT_CENTER shows the whole frame (letterboxed) so the skeleton, placed in the same
      // aspect-fit rectangle, lines up with the person.
      scaleType = PreviewView.ScaleType.FIT_CENTER
      implementationMode = PreviewView.ImplementationMode.COMPATIBLE
    }
  }

  LaunchedEffect(lensFacing, lifecycleOwner) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    cameraProviderFuture.addListener(
      {
        val cameraProvider = cameraProviderFuture.get()
        try {
          cameraProvider.unbindAll()
          bindCameraUseCases(
            lifecycleOwner = lifecycleOwner,
            cameraProvider = cameraProvider,
            executor = cameraExecutor,
            previewView = previewView,
            onImageAnalyzed = onImageAnalyzed,
            lensFacing = lensFacing,
          )
        } catch (exc: Exception) {
          Log.e(TAG, "Use case binding failed", exc)
        }
      },
      ContextCompat.getMainExecutor(context),
    )
  }

  DisposableEffect(lifecycleOwner) { onDispose { cameraProviderFuture.get().unbindAll() } }

  AndroidView(modifier = modifier, factory = { previewView })
}

fun bindCameraUseCases(
  lifecycleOwner: LifecycleOwner,
  cameraProvider: ProcessCameraProvider,
  executor: ExecutorService,
  previewView: PreviewView,
  onImageAnalyzed: (ImageProxy) -> Unit,
  lensFacing: Int,
) {
  val resolutionSelector =
    ResolutionSelector.Builder()
      .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
      .build()

  val preview: Preview = Preview.Builder().build()
  preview.setSurfaceProvider(previewView.surfaceProvider)

  val cameraSelector: CameraSelector =
    CameraSelector.Builder().requireLensFacing(lensFacing).build()
  val imageAnalysis =
    ImageAnalysis.Builder()
      .setResolutionSelector(resolutionSelector)
      .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
      .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
      .build()

  imageAnalysis.setAnalyzer(executor) { imageProxy -> onImageAnalyzed(imageProxy) }

  cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, imageAnalysis, preview)
}
