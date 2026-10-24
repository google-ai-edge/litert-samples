package com.example.qnn_litertlm_gemma

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class QualcommChatbotUnitTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun resolveModelInDirectories_prefersHuggingFaceFilenameAndSupportsLegacyFallback() {
        val externalDir = tempFolder.newFolder("external")
        val internalDir = tempFolder.newFolder("internal")

        // Empty file must be ignored
        File(externalDir, "gemma-4-E2B-it_qualcomm_sm8750.litertlm").writeBytes(ByteArray(0))
        assertNull(
            ModelDownloader.resolveModelInDirectories(
                ModelDownloader.GEMMA4_NPU,
                listOf(externalDir, internalDir)
            )
        )

        // Legacy model.litertlm fallback works seamlessly
        val legacyFile = File(externalDir, "model.litertlm").apply {
            writeBytes(byteArrayOf(1, 2, 3, 4))
        }
        assertEquals(
            legacyFile.canonicalPath,
            ModelDownloader.resolveModelInDirectories(
                ModelDownloader.GEMMA4_NPU,
                listOf(externalDir, internalDir)
            )?.canonicalPath
        )

        // Official SM8750 filename takes priority once non-empty
        val officialFile = File(externalDir, "gemma-4-E2B-it_qualcomm_sm8750.litertlm").apply {
            writeBytes(byteArrayOf(10, 20, 30, 40))
        }
        assertEquals(
            officialFile.canonicalPath,
            ModelDownloader.resolveModelInDirectories(
                ModelDownloader.GEMMA4_NPU,
                listOf(externalDir, internalDir)
            )?.canonicalPath
        )
    }

    @Test
    fun orderBackendNames_prioritizesRequestedBackendWhilePreservingFallbackOrder() {
        val defaultOrder = listOf("NPU", "GPU", "CPU")
        assertEquals(listOf("NPU", "GPU", "CPU"), LiteRTLMManager.orderBackendNames(defaultOrder, null))
        assertEquals(listOf("GPU", "NPU", "CPU"), LiteRTLMManager.orderBackendNames(defaultOrder, "gpu"))
        assertEquals(listOf("CPU", "NPU", "GPU"), LiteRTLMManager.orderBackendNames(defaultOrder, "CPU"))
        assertEquals(defaultOrder, LiteRTLMManager.orderBackendNames(defaultOrder, "UNKNOWN"))
    }

    @Test
    fun errorClassifiers_detectNpuKvCacheAndTemplateAlternationFailures() {
        val kvError = RuntimeException(
            "Status Code: 11. Message: ERROR: [llm_litert_npu_compiled_model_executor_utils.cc:369] KV-cache update out of range"
        )
        val templateError = RuntimeException(
            "INTERNAL: Failed to apply template: invalid operation: Conversation roles must alternate user/assistant/user/assistant/..."
        )
        val genericError = RuntimeException("Disk read failure")

        assertTrue(LiteRTLMManager.isContextOverflowError(kvError))
        assertFalse(LiteRTLMManager.isContextOverflowError(genericError))

        assertTrue(LiteRTLMManager.isRoleAlternationError(templateError))
        assertFalse(LiteRTLMManager.isRoleAlternationError(genericError))
        assertNotNull(ModelDownloader.FASTVLM_NPU)
    }
}
