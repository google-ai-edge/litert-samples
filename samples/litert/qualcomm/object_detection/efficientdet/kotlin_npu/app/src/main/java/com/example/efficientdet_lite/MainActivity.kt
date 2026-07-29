package com.example.efficientdet_lite

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.example.efficientdet_lite.ui.theme.EfficientDETLiteTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            EfficientDETLiteTheme {
                EfficientDetCameraScreen()
            }
        }
    }
}
