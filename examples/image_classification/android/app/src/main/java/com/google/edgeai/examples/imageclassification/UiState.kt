package com.google.edgeai.examples.imageclassification

import androidx.compose.runtime.Immutable

@Immutable
class UiState(
    val inferenceTime: Long = 0L,
    val categories: List<ImageClassificationHelper.Category> = emptyList(),
    val setting: Setting = Setting(),
    val errorMessage: String? = null,
)

@Immutable
data class Setting(
    val model: ImageClassificationHelper.Model = ImageClassificationHelper.DEFAULT_MODEL,
    val delegate: ImageClassificationHelper.Delegate = ImageClassificationHelper.DEFAULT_DELEGATE,
    val resultCount: Int = ImageClassificationHelper.DEFAULT_RESULT_COUNT,
    val threshold: Float = ImageClassificationHelper.DEFAULT_THRESHOLD,
    val threadCount: Int = ImageClassificationHelper.DEFAULT_THREAD_COUNT
)