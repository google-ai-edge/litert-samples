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

package com.google.ai.edge.examples.math_to_latex

import android.annotation.SuppressLint
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import org.json.JSONObject

/**
 * The answer drawn by MathJax in a WebView (assets/math.html). The page loads only the app's
 * assets: network loads are blocked. [onRendered] gets the number of formulas MathJax could not
 * parse.
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun MathView(text: String, modifier: Modifier = Modifier, onRendered: (errors: Int) -> Unit = {}) {
  var heightPx by remember { mutableIntStateOf(0) }
  val state = remember { MathPageState() }
  AndroidView(
    modifier = modifier.fillMaxWidth().height(if (heightPx > 0) (heightPx).dp else 1.dp),
    factory = { context ->
      WebView(context).apply {
        settings.javaScriptEnabled = true
        settings.blockNetworkLoads = true
        settings.allowContentAccess = false
        isVerticalScrollBarEnabled = false
        addJavascriptInterface(
          object {
            @JavascriptInterface
            fun rendered(height: Int, errors: Int) {
              Handler(Looper.getMainLooper()).post {
                heightPx = height
                Log.i(MathViewModel.TAG, "RENDERED height=$height errors=$errors")
                onRendered(errors)
              }
            }
          },
          "Page",
        )
        webViewClient =
          object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String?) {
              state.loaded = true
              state.pending?.let {
                view.evaluateJavascript("render(${JSONObject.quote(it)})", null)
              }
              state.pending = null
            }
          }
        loadUrl("file:///android_asset/math.html")
      }
    },
    update = { view ->
      // An empty text keeps the page loaded and hidden (1 dp high) until there is an answer to
      // draw, and
      // clears the last answer's drawing when a new read starts.
      if (text.isEmpty()) {
        if (state.shown != null) {
          state.shown = null
          heightPx = 0
          view.evaluateJavascript("document.getElementById('out').textContent = ''", null)
        }
      } else if (state.shown != text) {
        state.shown = text
        if (state.loaded) {
          view.evaluateJavascript("render(${JSONObject.quote(text)})", null)
        } else {
          state.pending = text
        }
      }
    },
  )
}

private class MathPageState {
  var loaded = false
  var pending: String? = null
  var shown: String? = null
}
