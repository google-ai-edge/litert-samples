package com.example.gemma3_on_device

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class Gemma3CpuGpuUnitTest {

    @Test
    fun `default Gemma 3 1B model config is valid and text-only`() {
        val defaultModel = ModelDownloader.AVAILABLE_MODELS.firstOrNull()
        assertNotNull(defaultModel)
        assertEquals("gemma3-1b-it-int4.litertlm", defaultModel!!.filename)
        assertEquals("GPU", defaultModel.preferredBackend)
        assertTrue(defaultModel.url.startsWith("https://huggingface.co/litert-community/Gemma3-1B-IT/"))
        assertFalse(defaultModel.systemPrompt.orEmpty().contains("images, and audio"))
    }

    @Test
    fun `context overflow and role alternation classifiers detect runtime exceptions`() {
        assertTrue(
            LiteRTLMManager.isContextOverflowError(
                RuntimeException("INVALID_ARGUMENT: KV-cache update out of range")
            )
        )
        assertTrue(
            LiteRTLMManager.isRoleAlternationError(
                RuntimeException("INVALID_ARGUMENT: Conversation roles must alternate user/assistant/")
            )
        )
        assertFalse(
            LiteRTLMManager.isContextOverflowError(RuntimeException("Network unreachable"))
        )
    }

    @Test
    fun `streamTurnWithOverflowRecovery resets session and retries prefill overflow`() = runBlocking {
        var attempts = 0
        var resets = 0
        val chunks = StringBuilder()

        val outcome = streamTurnWithOverflowRecovery(
            streamOnce = { emit ->
                attempts += 1
                if (attempts == 1) {
                    throw RuntimeException("KV-cache update out of range")
                }
                emit("Recovered ")
                emit("answer.")
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
        assertEquals("Recovered answer.", chunks.toString())
    }

    @Test
    fun `streamTurnWithOverflowRecovery preserves partial output on mid-stream overflow`() = runBlocking {
        var attempts = 0
        var resets = 0
        val chunks = StringBuilder()

        val outcome = streamTurnWithOverflowRecovery(
            streamOnce = { emit ->
                attempts += 1
                emit("Partial output before max context")
                throw RuntimeException("Reached max_num_tokens limit")
            },
            currentOutput = { chunks.toString() },
            resetSession = {
                resets += 1
                true
            },
            onChunk = { chunks.append(it) }
        )

        assertFalse(outcome.retriedInFreshSession)
        assertTrue(outcome.truncatedMidStream)
        assertEquals(1, attempts)
        assertEquals(1, resets)
        assertEquals("Partial output before max context", chunks.toString())
    }
}
