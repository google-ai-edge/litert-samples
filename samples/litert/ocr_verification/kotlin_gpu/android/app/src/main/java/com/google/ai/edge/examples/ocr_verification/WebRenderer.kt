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

package com.google.ai.edge.examples.ocr_verification

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.view.ViewGroup
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.webkit.WebViewAssetLoader
import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt
import org.json.JSONObject

/**
 * Draws an OCR text the way the source's renderer does, in a WebView (Chromium) laid out at the
 * source's 1600 x 1200 CSS-pixel viewport: the page of [RenderPage], then the capture area drawn at
 * the source's scale (1 pixel per CSS pixel for text and tables, 3 for formulas) and its white border
 * trimmed to 16 pixels. The WebView sits behind the app's screen; it is laid out but never seen.
 *
 * [WebView.enableSlowWholeDocumentDraw] must be called before the first WebView of the process is
 * made (MainActivity does), so that [WebView.draw] draws the parts outside the window too.
 */
@SuppressLint("SetJavaScriptEnabled")
class WebRenderer(context: Context, parent: ViewGroup) {
  class Rendered(val rgb: HostMath.Rgb, val errors: List<String>, val ms: Double, val cssWidth: Double)

  private val density = context.resources.displayMetrics.density
  private val main = Handler(Looper.getMainLooper())
  private val web = WebView(context)
  // the page and its KaTeX and font files come from the app's assets over one https origin
  private val assets =
    WebViewAssetLoader.Builder()
      .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(context))
      .build()
  private var pending: ((String?, String?) -> Unit)? = null
  private var loaded: (() -> Unit)? = null

  init {
    web.settings.javaScriptEnabled = true
    web.settings.useWideViewPort = true
    web.settings.loadWithOverviewMode = false
    web.settings.textZoom = 100
    web.settings.setSupportZoom(false)
    web.settings.allowFileAccess = false
    web.settings.allowContentAccess = false
    web.setBackgroundColor(Color.WHITE)
    web.isVerticalScrollBarEnabled = false
    web.isHorizontalScrollBarEnabled = false
    web.addJavascriptInterface(Bridge(), "OcrBridge")
    web.webViewClient =
      object : WebViewClient() {
        override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
          assets.shouldInterceptRequest(request.url)

        override fun onPageFinished(view: WebView?, url: String?) {
          loaded?.let {
            loaded = null
            it()
          }
        }
      }
    val lp =
      FrameLayout.LayoutParams(
        (RenderPage.VIEWPORT_WIDTH * density).roundToInt(),
        (RenderPage.VIEWPORT_HEIGHT * density).roundToInt(),
      )
    parent.addView(web, 0, lp)
  }

  private inner class Bridge {
    @JavascriptInterface
    fun done(json: String) {
      main.post { pending?.invoke(json, null) }
    }

    @JavascriptInterface
    fun failed(message: String) {
      main.post { pending?.invoke(null, message) }
    }
  }

  /**
   * Renders [text] as [category] for a region [targetWidth] pixels wide; [done] gets the picture or
   * the error. Main thread only, one request at a time.
   */
  fun render(
    text: String,
    category: RenderPage.Category,
    targetWidth: Int,
    done: (Result<Rendered>) -> Unit,
  ) {
    check(Looper.myLooper() == Looper.getMainLooper()) { "render on the main thread" }
    val t0 = System.nanoTime()
    val scale = category.scale
    fun fail(e: Throwable) = done(Result.failure(e))
    load(RenderPage.page(RenderPage.content(text, category))) {
      run(targetWidth) { json, error ->
        if (json == null) {
          fail(IllegalStateException("render script failed: $error"))
          return@run
        }
        val box = JSONObject(json)
        val errors = box.getJSONArray("errors").let { a -> List(a.length()) { a.getString(it) } }
        if (errors.isEmpty()) {
          capture(box, scale, trim = true) { rgb ->
            done(Result.success(Rendered(rgb, errors, (System.nanoTime() - t0) / 1e6, box.getDouble("width"))))
          }
        } else {
          // the source's ParseError path: the picture is the error text itself, not trimmed
          load(RenderPage.errorPage(errors)) {
            run(targetWidth) { errJson, err2 ->
              if (errJson == null) {
                fail(IllegalStateException("error page failed: $err2"))
                return@run
              }
              val errBox = JSONObject(errJson)
              capture(errBox, scale, trim = false) { rgb ->
                done(Result.success(Rendered(rgb, errors, (System.nanoTime() - t0) / 1e6, errBox.getDouble("width"))))
              }
            }
          }
        }
      }
    }
  }

  private fun load(html: String, then: () -> Unit) {
    loaded = then
    web.loadDataWithBaseURL(BASE_URL, html, "text/html", "utf-8", null)
  }

  private fun run(targetWidth: Int, then: (String?, String?) -> Unit) {
    pending = { json, error ->
      pending = null
      then(json, error)
    }
    web.evaluateJavascript("renderAll($targetWidth);", null)
  }

  /**
   * Draws the capture area's box (CSS pixels, enlarged to whole pixels as Playwright's element
   * screenshot does) at [scale] pixels per CSS pixel, once the page's last change is on screen.
   */
  private fun capture(box: JSONObject, scale: Float, trim: Boolean, then: (HostMath.Rgb) -> Unit) {
    val x = box.getDouble("x")
    val y = box.getDouble("y")
    val x0 = floor(x + 1e-3).toInt()
    val y0 = floor(y + 1e-3).toInt()
    val x1 = ceil(x + box.getDouble("width") - 1e-3).toInt()
    val y1 = ceil(y + box.getDouble("height") - 1e-3).toInt()
    val w = ((x1 - x0) * scale).roundToInt().coerceAtLeast(1)
    val h = ((y1 - y0) * scale).roundToInt().coerceAtLeast(1)
    web.postVisualStateCallback(
      System.nanoTime(),
      object : WebView.VisualStateCallback() {
        override fun onComplete(requestId: Long) {
          val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
          bmp.eraseColor(Color.WHITE)
          val canvas = Canvas(bmp)
          // the WebView draws 1 CSS pixel as `density` view pixels; the picture wants `scale`
          canvas.scale(scale / density, scale / density)
          canvas.translate(-x0 * density, -y0 * density)
          web.draw(canvas)
          val rgb = toRgb(bmp)
          bmp.recycle()
          then(if (trim) trimWhite(rgb, RenderPage.TRIM_PADDING) else rgb)
        }
      },
    )
  }

  fun destroy() {
    (web.parent as? ViewGroup)?.removeView(web)
    web.destroy()
  }

  companion object {
    const val BASE_URL = "https://appassets.androidplatform.net/assets/render/"

    fun toRgb(bmp: Bitmap): HostMath.Rgb {
      val w = bmp.width
      val h = bmp.height
      val buf = ByteBuffer.allocate(w * h * 4)
      bmp.copyPixelsToBuffer(buf)
      val rgba = buf.array()
      val out = ByteArray(w * h * 3)
      for (i in 0 until w * h) {
        out[i * 3] = rgba[i * 4]
        out[i * 3 + 1] = rgba[i * 4 + 1]
        out[i * 3 + 2] = rgba[i * 4 + 2]
      }
      return HostMath.Rgb(out, w, h)
    }

    /** render_label._trim_white_border: the box of non-white pixels, [padding] kept around it. */
    fun trimWhite(rgb: HostMath.Rgb, padding: Int): HostMath.Rgb {
      var left = rgb.width
      var top = rgb.height
      var right = -1
      var bottom = -1
      val b = rgb.bytes
      for (y in 0 until rgb.height) {
        for (x in 0 until rgb.width) {
          val o = (y * rgb.width + x) * 3
          if (b[o].toInt() != -1 || b[o + 1].toInt() != -1 || b[o + 2].toInt() != -1) {
            if (x < left) left = x
            if (x > right) right = x
            if (y < top) top = y
            if (y > bottom) bottom = y
          }
        }
      }
      if (right < 0) return rgb
      val l = maxOf(left - padding, 0)
      val t = maxOf(top - padding, 0)
      val r = minOf(right + 1 + padding, rgb.width)
      val btm = minOf(bottom + 1 + padding, rgb.height)
      return crop(rgb, l, t, r - l, btm - t)
    }

    fun crop(rgb: HostMath.Rgb, x: Int, y: Int, w: Int, h: Int): HostMath.Rgb {
      val out = ByteArray(w * h * 3)
      for (row in 0 until h) {
        System.arraycopy(rgb.bytes, ((y + row) * rgb.width + x) * 3, out, row * w * 3, w * 3)
      }
      return HostMath.Rgb(out, w, h)
    }
  }
}
