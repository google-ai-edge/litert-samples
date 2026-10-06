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

package com.example.litertruntime

import android.app.Application
import android.content.Context
import android.content.res.Configuration
import android.content.res.loader.ResourcesLoader
import android.content.res.loader.ResourcesProvider
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File

/** An [Application] that reads its assets from [base]. The framework does not start it. */
class FixtureApplication(base: Context) : Application() {
    init {
        attachBaseContext(base)
    }
}

/**
 * Serves `assets/model.tflite` from a directory the test rewrites. The skill's `Classifier` always
 * opens that one asset, so this is how a test hands it a model that makes one step throw, and how
 * one ViewModel loads a different model on its next `load()`.
 *
 * Uses a [ResourcesLoader], so the tests need Android 11 (API 30) or later.
 */
class ModelAssets {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val dir = File(instrumentation.targetContext.cacheDir, "model-assets")

    /** Pass this to the ViewModel in place of the real application. */
    val application: Application

    init {
        dir.deleteRecursively()
        File(dir, "assets").mkdirs()
        use(APP_MODEL)
        val context = instrumentation.targetContext.createConfigurationContext(Configuration())
        val loader = ResourcesLoader()
        loader.addProvider(ResourcesProvider.loadFromDirectory(dir.path, null))
        context.resources.addLoaders(loader)
        application = FixtureApplication(context)
    }

    /**
     * Makes [fixture] the model the next load reads: [APP_MODEL], or the name of a file in
     * `fixtures` without `.bin`.
     */
    fun use(fixture: String) {
        val source =
            if (fixture == APP_MODEL) {
                instrumentation.targetContext.assets.open("model.tflite")
            } else {
                instrumentation.context.assets.open("$fixture.bin")
            }
        source.use { input ->
            File(dir, "assets/model.tflite").outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }

    companion object {
        /** The model the app ships as `assets/model.tflite`. */
        const val APP_MODEL = "model"
    }
}
