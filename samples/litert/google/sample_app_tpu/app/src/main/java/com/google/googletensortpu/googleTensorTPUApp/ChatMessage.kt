package com.google.googletensortpu.googleTensorTPUApp

import android.graphics.Bitmap

data class ChatMessage(
    val text: String,
    val isUser: Boolean,
    val image: Bitmap? = null,
    val audioPath: String? = null,
    val imagePath: String? = null
)