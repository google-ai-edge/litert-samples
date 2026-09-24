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

package com.google.ai.edge.examples.model_zoo

import android.graphics.Bitmap
import com.google.ai.edge.examples.model_zoo.audio.AudioStageProgress
import com.google.ai.edge.examples.model_zoo.data.DownloadConfirmation
import com.google.ai.edge.examples.model_zoo.data.DownloadState
import com.google.ai.edge.examples.model_zoo.data.ModelEntry
import com.google.ai.edge.examples.model_zoo.models.rfdetr.RfDetr

/** UI snapshots; model objects and buffers stay inside the ViewModel worker. */
data class UiState(
  val tasks: List<ModelEntry> = emptyList(),
  val downloads: Map<String, DownloadState> = emptyMap(),
  val downloadConfirmation: DownloadConfirmation? = null,
  val screen: String = "home",
  val selectedTaskId: String? = null,
  val busy: Boolean = false,
  val loading: Boolean = true,
  val error: String? = null,
  val storageBytes: Long = 0,
  val inputText: String = "Hello. This speech is generated on your device.",
  val image: Bitmap? = null,
  val boxes: List<RfDetr.Detection> = emptyList(),
  val labels: List<String> = emptyList(),
  val inferenceMs: Double? = null,
  val backend: String = "",
  val fallbackReason: String? = null,
  val backendDetails: String = "",
  val audioReady: Boolean = false,
  val speechExportReady: Boolean = false,
  val speechSaving: Boolean = false,
  val speechSaved: Boolean = false,
  val playing: Boolean = false,
  val playbackElapsedSeconds: Float = 0f,
  val playbackTotalSeconds: Float = 0f,
  val camera: Boolean = false,
  val recording: Boolean = false,
  val recordedSeconds: Float = 0f,
  val audioInputName: String? = null,
  val audioInputSeconds: Float? = null,
  val transcript: String? = null,
  val inputImage: Bitmap? = null,
  val outputImage: Bitmap? = null,
  val imageOutputText: String = "",
  val imageOutputDetails: String = "",
  val secondaryImage: Bitmap? = null,
  val cameraFrames: Int = 0,
  val audioOutputs: List<String> = emptyList(),
  val audioSummary: String = "",
  val audioProgress: AudioStageProgress? = null,
  val pitchHz: List<Float> = emptyList(),
  val pitchConfidence: List<Float> = emptyList(),
  val pitchHopSeconds: Float = 0.1f,
)
