package com.example.validation

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

// LiteRT CompiledModel API
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

@RunWith(AndroidJUnit4::class)
class MigrationValidationTest {

    @Test
    fun verifyMigrationInference() {
        val context = ApplicationProvider.getApplicationContext<Context>()

        // TODO: Replace with the actual model name in the assets folder
        val modelAssetPath = "model.tflite"

        // Initialize CompiledModel from assets on the CPU (the like-for-like baseline).
        val compiledModel = CompiledModel.create(
            context.assets,
            modelAssetPath,
            CompiledModel.Options(Accelerator.CPU)
        )

        // Buffers are created by the model with the right sizes; create once, reuse, close.
        val inputs = compiledModel.createInputBuffers()
        val outputs = compiledModel.createOutputBuffers()

        // TODO: Replace with a real preprocessed input; the element count must match the model.
        val inputSize = 100 // replace with actual size
        inputs[0].writeFloat(FloatArray(inputSize) { 0.5f })

        // Run inference; readFloat() waits for the result.
        compiledModel.run(inputs, outputs)
        val outputArray = outputs[0].readFloat()

        // Verify output is populated (non-zero or changed from initial state)
        val isPopulated = outputArray.any { it != 0.0f }

        inputs.forEach { it.close() }
        outputs.forEach { it.close() }
        compiledModel.close()

        assertTrue("Output buffer should have non-zero results from inference", isPopulated)
    }
}
