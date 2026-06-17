package com.google.edgetpu.edgeTPUApp

import android.content.Context
import android.os.Bundle
import android.widget.Spinner
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import org.junit.Assert.assertTrue

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [30])
class MainActivityTest {

    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        
        // Reset AVAILABLE_MODELS to initial state
        ModelResolver.AVAILABLE_MODELS.clear()
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.GEMMA3_CPU)
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.TINY_GARDEN)
        
        val prefs = context.getSharedPreferences("ModelResolverPrefs", Context.MODE_PRIVATE)
        prefs.edit().clear().commit()
    }

    @Test
    fun testGetConversationHistoryMessages() {
        val controller = Robolectric.buildActivity(MainActivity::class.java).setup()
        val activity = controller.get()
        
        // Let's populate messagesList using reflection
        val messagesListField = MainActivity::class.java.getDeclaredField("messagesList")
        messagesListField.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        val messagesList = messagesListField.get(activity) as MutableList<ChatMessage>
        
        // System message (should be ignored)
        messagesList.add(ChatMessage("Welcome system message", false))
        
        // User turn 1
        messagesList.add(ChatMessage("Hello model", true))
        
        // Model turn 1
        messagesList.add(ChatMessage("Hello user!", false))
        
        // User turn 2 (multimodal - fake files)
        val imageFile = File(activity.cacheDir, "fake_image.jpg")
        imageFile.createNewFile()
        val audioFile = File(activity.cacheDir, "fake_audio.wav")
        audioFile.createNewFile()
        
        messagesList.add(
            ChatMessage(
                text = "See this and hear this",
                isUser = true,
                imagePath = imageFile.absolutePath,
                audioPath = audioFile.absolutePath
            )
        )
        
        // Model turn 2
        messagesList.add(ChatMessage("I see and hear it.", false))
        
        // Model turn 3 (system message or error should be ignored)
        messagesList.add(ChatMessage("Error: something failed", false))
        
        // Invoke getConversationHistoryMessages via reflection
        val getHistoryMethod = MainActivity::class.java.getDeclaredMethod("getConversationHistoryMessages")
        getHistoryMethod.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        val history = getHistoryMethod.invoke(activity) as List<com.google.ai.edge.litertlm.Message>
        
        // Check size: should be 4 (User 1, Model 1, User 2, Model 2)
        assertEquals(4, history.size)
        
        // Turn 1
        assertEquals(com.google.ai.edge.litertlm.Role.USER, history[0].role)
        assertEquals("Hello model", history[0].contents.contents[0].let { (it as com.google.ai.edge.litertlm.Content.Text).text })
        
        // Turn 2
        assertEquals(com.google.ai.edge.litertlm.Role.MODEL, history[1].role)
        assertEquals("Hello user!", history[1].contents.contents[0].let { (it as com.google.ai.edge.litertlm.Content.Text).text })
        
        // Turn 3
        assertEquals(com.google.ai.edge.litertlm.Role.USER, history[2].role)
        val turn3Contents = history[2].contents.contents
        assertEquals(3, turn3Contents.size)
        assertTrue(turn3Contents[0] is com.google.ai.edge.litertlm.Content.ImageFile)
        assertTrue(turn3Contents[1] is com.google.ai.edge.litertlm.Content.AudioFile)
        assertEquals("See this and hear this", (turn3Contents[2] as com.google.ai.edge.litertlm.Content.Text).text)
        
        // Turn 4
        assertEquals(com.google.ai.edge.litertlm.Role.MODEL, history[3].role)
        assertEquals("I see and hear it.", history[3].contents.contents[0].let { (it as com.google.ai.edge.litertlm.Content.Text).text })
    }

    @Test
    fun testActivityRecreationRestoresCustomModel() {
        // 1. Create a custom model and save it
        val customModel = ModelConfig(
            id = "my-custom-model",
            name = "My Custom Model",
            filename = "my_custom.litertlm",
            isAsset = false
        )
        ModelResolver.saveCustomModel(context, customModel)

        // 2. Launch the activity using Robolectric
        val controller = Robolectric.buildActivity(MainActivity::class.java)
        controller.create()
        controller.start()
        controller.resume()
        
        val activity = controller.get()
        org.robolectric.shadows.ShadowLooper.idleMainLooper()
        
        // Let's programmatically select the custom model in the spinner/selection
        val spinner: Spinner = activity.findViewById(R.id.spinnerModel)
        
        // Make sure custom model exists in the adapter
        val adapter = spinner.adapter
        assertNotNull(adapter)
        
        // Trigger listener manually
        val listener = spinner.onItemSelectedListener
        assertNotNull(listener)
        // First call to disable isInitializing
        listener!!.onItemSelected(spinner, null, 0, 0L)
        // Second call to select the custom model
        val customModelIndex = ModelResolver.AVAILABLE_MODELS.indexOfFirst { it.id == "my-custom-model" } + 1
        listener.onItemSelected(spinner, null, customModelIndex, 0L)
        
        val initialSelectedModelField = MainActivity::class.java.getDeclaredField("selectedModel")
        initialSelectedModelField.isAccessible = true
        val initialSelectedModel = initialSelectedModelField.get(activity) as? ModelConfig
        assertNotNull("selectedModel should be set on initial activity before saving state", initialSelectedModel)
        assertEquals("my-custom-model", initialSelectedModel?.id)
        
        // Save instance state
        val savedInstanceState = Bundle()
        controller.saveInstanceState(savedInstanceState)
        
        // Destroy the activity
        controller.pause()
        controller.stop()
        controller.destroy()

        // Simulate process death / classloader reload by resetting static state
        ModelResolver.AVAILABLE_MODELS.clear()
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.GEMMA3_CPU)
        ModelResolver.AVAILABLE_MODELS.add(ModelResolver.TINY_GARDEN)

        // 3. Recreate the activity with the saved instance state
        val recreatedController = Robolectric.buildActivity(MainActivity::class.java)
        recreatedController.create(savedInstanceState)
        recreatedController.start()
        recreatedController.restoreInstanceState(savedInstanceState)
        recreatedController.resume()
        recreatedController.visible()
        
        val recreatedActivity = recreatedController.get()
        
        // 4. Verify that selectedModel is restored and not null using reflection
        val selectedModelField = MainActivity::class.java.getDeclaredField("selectedModel")
        selectedModelField.isAccessible = true
        val restoredModel = selectedModelField.get(recreatedActivity) as? ModelConfig
        
        assertNotNull("selectedModel should not be null after recreation", restoredModel)
        assertEquals("my-custom-model", restoredModel?.id)
    }

}
