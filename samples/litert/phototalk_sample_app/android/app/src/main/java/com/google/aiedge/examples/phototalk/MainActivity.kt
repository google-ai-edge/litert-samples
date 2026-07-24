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

package com.google.aiedge.examples.phototalk

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.google.aiedge.examples.phototalk.ui.PhotoTalkAppScreen

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        checkAndRequestStoragePermissions()
        setContent {
            MaterialTheme {
                val uiState by viewModel.uiState.collectAsState()
                PhotoTalkAppScreen(
                    uiState = uiState,
                    onImageSelected = { uri -> viewModel.onImageSelected(uri) },
                    onModelPathChanged = { path -> viewModel.updateModelPath(path) },
                    onModelUriPicked = { uri -> viewModel.onModelUriPicked(uri) },
                    onBackendChanged = { backend -> viewModel.updatePreferredBackend(backend) },
                    onInitLmEngine = { viewModel.initializeLmEngine() },
                    onSendMessage = { text -> viewModel.sendMessage(text) }
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        viewModel.scanAvailableModelFiles()
    }

    private fun checkAndRequestStoragePermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                try {
                    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                        data = Uri.parse("package:$packageName")
                    }
                    startActivity(intent)
                } catch (e: Exception) {
                    val intent = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                    startActivity(intent)
                }
            }
        }
    }
}
