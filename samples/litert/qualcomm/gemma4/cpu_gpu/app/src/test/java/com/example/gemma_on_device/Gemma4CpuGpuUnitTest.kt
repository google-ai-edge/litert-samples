package com.example.gemma_on_device

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files

class Gemma4CpuGpuUnitTest {

    @Test
    fun `default Gemma 4 E2B model config matches multimodal litertlm package`() {
        val defaultModel = ModelDownloader.AVAILABLE_MODELS.firstOrNull()
        assertNotNull(defaultModel)
        assertEquals("gemma-4-E2B-it.litertlm", defaultModel!!.filename)
        assertEquals("GPU", defaultModel.preferredBackend)
        assertTrue(
            defaultModel.url.startsWith(
                "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/"
            )
        )
    }

    @Test
    fun `writeWavHeader produces valid 44-byte 16kHz mono 16-bit PCM RIFF WAVE header`() {
        val tempWav: File = Files.createTempFile("gemma4_audio_test_", ".wav").toFile()
        try {
            val pcmPayloadBytes = 32_000L // 1 second of 16 kHz mono 16-bit PCM
            RandomAccessFile(tempWav, "rw").use { raf ->
                writeWavHeader(
                    wav = raf,
                    sampleRate = 16_000,
                    channels = 1,
                    bitsPerSample = 16,
                    pcmBytes = pcmPayloadBytes
                )
            }

            val header = tempWav.readBytes()
            assertEquals(44, header.size)
            assertArrayEquals("RIFF".toByteArray(), header.sliceArray(0..3))
            assertArrayEquals("WAVE".toByteArray(), header.sliceArray(8..11))
            assertArrayEquals("fmt ".toByteArray(), header.sliceArray(12..15))
            assertArrayEquals("data".toByteArray(), header.sliceArray(36..39))

            val le = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
            assertEquals(36 + pcmPayloadBytes.toInt(), le.getInt(4))
            assertEquals(16, le.getInt(16))
            assertEquals(1.toShort(), le.getShort(20)) // PCM format = 1
            assertEquals(1.toShort(), le.getShort(22)) // Mono = 1 channel
            assertEquals(16_000, le.getInt(24)) // Sample rate
            assertEquals(32_000, le.getInt(28)) // Byte rate = 16000 * 1 * 2
            assertEquals(2.toShort(), le.getShort(32)) // Block align = 2 bytes
            assertEquals(16.toShort(), le.getShort(34)) // 16 bits/sample
            assertEquals(pcmPayloadBytes.toInt(), le.getInt(40))
        } finally {
            tempWav.delete()
        }
    }

    @Test
    fun `streamTurnWithOverflowRecovery recovers from role alternation or KV overflow`() = runBlocking {
        var attempts = 0
        var resets = 0
        val chunks = StringBuilder()

        val outcome = streamTurnWithOverflowRecovery(
            streamOnce = { emit ->
                attempts += 1
                if (attempts == 1) {
                    throw RuntimeException("Conversation roles must alternate user/assistant/")
                }
                emit("Multimodal session recovered.")
            },
            currentOutput = { chunks.toString() },
            resetSession = {
                resets += 1
                true
            },
            onChunk = { chunks.append(it) }
        )

        assertTrue(outcome.retriedInFreshSession)
        assertFalse(outcome.truncatedMidStream)
        assertEquals(2, attempts)
        assertEquals(1, resets)
        assertEquals("Multimodal session recovered.", chunks.toString())
    }
}
