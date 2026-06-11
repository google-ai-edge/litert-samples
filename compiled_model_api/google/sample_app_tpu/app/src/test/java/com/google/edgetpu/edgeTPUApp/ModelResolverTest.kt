package com.google.edgetpu.edgeTPUApp

import android.content.Context
import android.content.SharedPreferences
import androidx.test.core.app.ApplicationProvider
import org.json.JSONArray
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.io.FileOutputStream

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [30])
class ModelResolverTest {

    private lateinit var context: Context
    private lateinit var modelResolver: ModelResolver
    private lateinit var sharedPreferences: SharedPreferences

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        modelResolver = ModelResolver(context)
        sharedPreferences = context.getSharedPreferences("ModelResolverPrefs", Context.MODE_PRIVATE)
        sharedPreferences.edit().clear().commit()
        
        // Reset AVAILABLE_MODELS to initial state
        ModelResolver.AVAILABLE_MODELS.clear()
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.GEMMA3_CPU)
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.TINY_GARDEN)
    }

    @After
    fun tearDown() {
        sharedPreferences.edit().clear().commit()
        ModelResolver.AVAILABLE_MODELS.clear()
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.GEMMA3_CPU)
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.TINY_GARDEN)
    }

    @Test
    fun testInitialAvailableModels() {
        assertEquals(2, ModelResolver.AVAILABLE_MODELS.size)
        assertEquals("gemma3-cpu", ModelResolver.AVAILABLE_MODELS[0].id)
        assertEquals("tiny-garden", ModelResolver.AVAILABLE_MODELS[1].id)
    }

    @Test
    fun testSaveAndLoadCustomModel() {
        val customModel = ModelConfig(
            id = "custom-id",
            name = "Custom Model",
            filename = "custom.litertlm",
            isAsset = false,
            systemPrompt = "Custom Prompt",
            preferredBackend = "GPU",
            supportsImage = true,
            supportsAudio = false,
            defaultPrompt = "Custom Default"
        )

        ModelResolver.saveCustomModel(context, customModel)
        
        // Verify it was added
        assertEquals(3, ModelResolver.AVAILABLE_MODELS.size)
        assertEquals(customModel, ModelResolver.AVAILABLE_MODELS[2])

        // Clear in-memory models to simulate app restart
        ModelResolver.AVAILABLE_MODELS.clear()
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.GEMMA3_CPU)
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.TINY_GARDEN)

        // Load custom models
        ModelResolver.loadCustomModels(context)

        // Verify it was loaded correctly
        assertEquals(3, ModelResolver.AVAILABLE_MODELS.size)
        val loadedModel = ModelResolver.AVAILABLE_MODELS[2]
        assertEquals(customModel.id, loadedModel.id)
        assertEquals(customModel.name, loadedModel.name)
        assertEquals(customModel.filename, loadedModel.filename)
        assertFalse(loadedModel.isAsset)
        assertEquals(customModel.systemPrompt, loadedModel.systemPrompt)
        assertEquals(customModel.preferredBackend, loadedModel.preferredBackend)
    }

    @Test
    fun testDeleteCustomModel() {
        val customModel = ModelConfig(
            id = "custom-id-delete",
            name = "Delete Model",
            filename = "delete.litertlm",
            isAsset = false
        )

        // Add model
        ModelResolver.saveCustomModel(context, customModel)
        assertEquals(3, ModelResolver.AVAILABLE_MODELS.size)

        // Create a fake file to ensure deletion logic works
        val externalDir = context.getExternalFilesDir(null)
        val file = File(externalDir, customModel.filename)
        file.createNewFile()
        assertTrue(file.exists())

        // Delete model
        ModelResolver.deleteCustomModel(context, customModel)

        assertEquals(2, ModelResolver.AVAILABLE_MODELS.size)
        assertFalse(file.exists())

        // Verify preferences were updated
        val json = sharedPreferences.getString("CustomModels", null)
        val array = JSONArray(json)
        assertEquals(0, array.length()) // No custom models left
    }

    @Test
    fun testIsModelAvailable_Asset() {
        assertTrue(modelResolver.isModelAvailable(ModelResolver.GEMMA3_CPU))
    }

    @Test
    fun testIsModelAvailable_External_NotExists() {
        val customModel = ModelConfig(
            id = "ext-id",
            name = "Ext",
            filename = "ext.litertlm",
            isAsset = false
        )
        assertFalse(modelResolver.isModelAvailable(customModel))
    }

    @Test
    fun testIsModelAvailable_External_ExistsInCache() {
        val customModel = ModelConfig(
            id = "ext-id",
            name = "Ext",
            filename = "ext.litertlm",
            isAsset = false
        )
        val cacheFile = File(context.cacheDir, customModel.filename)
        FileOutputStream(cacheFile).use { it.write("fake model content".toByteArray()) }

        assertTrue(modelResolver.isModelAvailable(customModel))
    }
}
