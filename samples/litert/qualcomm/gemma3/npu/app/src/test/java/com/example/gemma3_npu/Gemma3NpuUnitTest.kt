package com.example.gemma3_npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class Gemma3NpuUnitTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun findReadableModel_ignoresZeroBytePartialFilesAndFindsValidPackage() {
        val internalDir = tempFolder.newFolder("internal")
        val externalDir = tempFolder.newFolder("external")
        val filename = ModelDownloader.AVAILABLE_MODELS.first().filename

        // Zero-byte partial file must not be treated as a valid downloaded model
        File(internalDir, filename).writeBytes(ByteArray(0))
        assertNull(ModelDownloader.findReadableModel(filename, listOf(internalDir, externalDir)))

        val validExternal = File(externalDir, filename).apply {
            writeBytes(byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8))
        }
        assertEquals(
            validExternal.canonicalPath,
            ModelDownloader.findReadableModel(filename, listOf(internalDir, externalDir))?.canonicalPath
        )
    }

    @Test
    fun orderBackendNames_prioritizesPreferredBackendWithDeterministicFallback() {
        val defaultOrder = listOf("NPU", "GPU", "CPU")
        assertEquals(listOf("NPU", "GPU", "CPU"), LiteRTLMManager.orderBackendNames(defaultOrder, "NPU"))
        assertEquals(listOf("GPU", "NPU", "CPU"), LiteRTLMManager.orderBackendNames(defaultOrder, "GPU"))
        assertEquals(listOf("CPU", "NPU", "GPU"), LiteRTLMManager.orderBackendNames(defaultOrder, "cpu"))
    }

    @Test
    fun errorClassifiers_detectIssue136KvCacheOverflowAndRoleAlternation() {
        val kvOverflow = RuntimeException(
            "Status Code: 11. Message: ERROR: [third_party/odml/litert_lm/runtime/executor/llm_litert_npu_compiled_model_executor_utils.cc:369] KV-cache update out of range"
        )
        val roleAlternation = RuntimeException(
            "Failed to start nativeSendMessageAsync: INTERNAL: Failed to apply template: invalid operation: Conversation roles must alternate user/assistant/user/assistant/... (in template:15)"
        )

        assertTrue(LiteRTLMManager.isContextOverflowError(kvOverflow))
        assertTrue(LiteRTLMManager.isRoleAlternationError(roleAlternation))
        assertFalse(LiteRTLMManager.isContextOverflowError(RuntimeException("Network timeout")))
    }
}
