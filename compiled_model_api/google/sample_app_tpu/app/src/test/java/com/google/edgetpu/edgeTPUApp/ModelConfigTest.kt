package com.google.edgetpu.edgeTPUApp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelConfigTest {

    @Test
    fun testModelConfigDefaults() {
        val config = ModelConfig(
            id = "test-id",
            name = "Test Model",
            filename = "test.litertlm",
            isAsset = true
        )

        assertEquals("test-id", config.id)
        assertEquals("Test Model", config.name)
        assertEquals("test.litertlm", config.filename)
        assertTrue(config.isAsset)
        
        // Defaults
        assertEquals(null, config.systemPrompt)
        assertEquals(null, config.preferredBackend)
        assertTrue(config.supportsImage)
        assertFalse(config.supportsAudio)
        assertEquals("Describe what you see.", config.defaultPrompt)
    }

    @Test
    fun testModelConfigCustomValues() {
        val config = ModelConfig(
            id = "test-id-custom",
            name = "Test Model Custom",
            filename = "test_custom.litertlm",
            isAsset = false,
            systemPrompt = "You are a custom assistant.",
            preferredBackend = "GPU",
            supportsImage = false,
            supportsAudio = true,
            defaultPrompt = "What is this sound?"
        )

        assertEquals("test-id-custom", config.id)
        assertEquals("Test Model Custom", config.name)
        assertEquals("test_custom.litertlm", config.filename)
        assertFalse(config.isAsset)
        assertEquals("You are a custom assistant.", config.systemPrompt)
        assertEquals("GPU", config.preferredBackend)
        assertFalse(config.supportsImage)
        assertTrue(config.supportsAudio)
        assertEquals("What is this sound?", config.defaultPrompt)
    }
}
