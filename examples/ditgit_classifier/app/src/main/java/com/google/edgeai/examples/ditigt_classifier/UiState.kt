package com.google.edgeai.examples.ditigt_classifier

import androidx.compose.runtime.Immutable

@Immutable
class UiState(
    val digit: String = "-",
    val score: Float = 0f,
    val drawOffsets: List<DrawOffset> = emptyList()
)

abstract class DrawOffset

data class Start(val x: Float, val y: Float) : DrawOffset()
data class Point(val x: Float, val y: Float) : DrawOffset()