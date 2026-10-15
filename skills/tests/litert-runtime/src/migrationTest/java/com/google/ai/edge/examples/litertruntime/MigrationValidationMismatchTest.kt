/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.litertruntime

import android.content.res.loader.ResourcesLoader
import android.content.res.loader.ResourcesProvider
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The migration gate has to fail on a golden output that the model does not reproduce. This test
 * serves `fixtures/golden/output_wrong.txt` (the golden with its first value moved by 0.01, ten
 * times the gate's bound) as the test APK's `golden/output.txt`, through a [ResourcesLoader] (API
 * 30), and runs [MigrationValidationTest], which is the skill's template with only the package
 * changed.
 */
@RunWith(AndroidJUnit4::class)
class MigrationValidationMismatchTest {
    @Test
    fun gateFailsOnAWrongGolden() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val dir = File(instrumentation.targetContext.cacheDir, "wrong-golden")
        dir.deleteRecursively()
        val golden = File(dir, "assets/golden/output.txt")
        checkNotNull(golden.parentFile).mkdirs()
        instrumentation.context.assets.open("golden/output_wrong.txt").use { input ->
            golden.outputStream().use { output -> input.copyTo(output) }
        }
        // The template reads the golden from the test APK's assets; the loader's file comes first.
        val loader = ResourcesLoader()
        loader.addProvider(ResourcesProvider.loadFromDirectory(dir.path, null))
        val resources = instrumentation.context.resources
        resources.addLoaders(loader)
        val error =
            try {
                assertThrows(AssertionError::class.java) {
                    MigrationValidationTest().verifyMigrationInference()
                }
            } finally {
                resources.removeLoaders(loader)
                dir.deleteRecursively()
            }
        Log.i(TAG, "The gate failed on the wrong golden: ${error.message}")
        assertTrue(
            "The gate failed for another reason: ${error.message}",
            error.message.orEmpty().startsWith("Output differs from the golden"),
        )
    }

    private companion object {
        const val TAG = "MigrationValidationMismatchTest"
    }
}
