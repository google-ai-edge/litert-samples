package com.google.edgetpu.edgeTPUApp

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [30])
class ChatMessageTest {

    @Test
    fun testChatMessageCreation() {
        val text = "Hello"
        val message = ChatMessage(text = text, isUser = true)

        assertEquals("Hello", message.text)
        assertTrue(message.isUser)
        assertNull(message.image)
        assertNull(message.audioPath)
    }

    @Test
    fun testChatMessageWithMedia() {
        val bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val message = ChatMessage(
            text = "Analyze this",
            isUser = true,
            image = bitmap,
            audioPath = "/path/to/audio.wav"
        )

        assertEquals("Analyze this", message.text)
        assertTrue(message.isUser)
        assertEquals(bitmap, message.image)
        assertEquals("/path/to/audio.wav", message.audioPath)
        assertNull(message.imagePath)
    }

    @Test
    fun testChatMessageWithImagePath() {
        val message = ChatMessage(
            text = "Analyze image path",
            isUser = true,
            imagePath = "/path/to/image.jpg"
        )
        assertEquals("Analyze image path", message.text)
        assertTrue(message.isUser)
        assertNull(message.image)
        assertNull(message.audioPath)
        assertEquals("/path/to/image.jpg", message.imagePath)
    }
}
