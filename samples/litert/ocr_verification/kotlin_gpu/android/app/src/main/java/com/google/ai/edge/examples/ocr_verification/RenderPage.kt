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

/**
 * The page the renderer draws an OCR text on, following the source model's renderer
 * (OpenBMB/MiniCPM-OCR, scripts/minicpm-ocr-jev/renderer/render_module/render_label.py, Apache-2.0,
 * render pipeline v8.5): one left-to-right pass that protects the math spans, the Markdown
 * conversion, the HTML path for tables, the page's CSS and its layout scripts. The model was trained
 * on pictures of this page, so a different page is a different input.
 *
 * Markdown: the source converts with Python-Markdown 3.10.3 (tables, fenced_code, nl2br). This file
 * covers what OCR text of a region needs (paragraphs, line breaks, emphasis, code spans, links,
 * backslash escapes, headings, bullet lists, inline HTML, the HTML escapes); other block syntax
 * (block quotes, code blocks, rules, Markdown tables) is drawn as plain text.
 */
object RenderPage {
  enum class Category(val label: String, val scale: Float) {
    TEXT("Text", 1f),
    FORMULA("Formula", 3f),
    TABLE("Table", 1f),
  }

  /** The viewport the source's browser context uses, in CSS pixels. */
  const val VIEWPORT_WIDTH = 1600
  const val VIEWPORT_HEIGHT = 1200

  /** White border kept around the ink when the picture is trimmed, in pixels. */
  const val TRIM_PADDING = 16

  // ------------------------------------------------------------ math spans

  private class Delim(val left: String, val right: String, val display: Boolean, val newline: Boolean)

  private val DELIMS =
    listOf(
      Delim("$$", "$$", true, true),
      Delim("\\[", "\\]", true, true),
      Delim("\\(", "\\)", false, true),
      Delim("$", "$", false, false),
    )

  private val STRONG_MATH =
    Regex(
      "(\\\\|[_^{}=<>+*/]|[A-Za-z]\\s*'+\\s*(?:\\(|[A-Za-z0-9])|" +
        "\\b(?:alpha|beta|gamma|theta|lambda|mu|pi|sigma|omega|sqrt|frac|sum|int|" +
        "lfloor|rfloor|mathrm|mathbb|operatorname|begin|end|times|cdot|leq|geq|pmod)\\b)"
    )
  private val PROSE_WORD =
    Regex("\\b(?:or|and|in|by|with|for|to|from|of|the|a|an)\\b", RegexOption.IGNORE_CASE)
  private val HTML_BOUNDARY_IN_MATH =
    Regex(
      "(?:<\\s*/?\\s*[A-Za-z][A-Za-z0-9:-]*(?:\\s+[^<>]*)?\\s*/?\\s*>|" +
        "&lt;\\s*/?\\s*[A-Za-z][A-Za-z0-9:-]*(?:\\s+[^&<>]*)?\\s*/?\\s*&gt;)",
      RegexOption.IGNORE_CASE,
    )

  private fun isEscaped(text: String, pos: Int): Boolean {
    var n = 0
    var i = pos - 1
    while (i >= 0 && text[i] == '\\') {
      n++
      i--
    }
    return n % 2 == 1
  }

  private class Span(val math: Boolean, val display: Boolean, val payload: String)

  /** KaTeX's splitAtDelimiters in one pass; a rejected span advances by its left delimiter only. */
  private fun splitMath(text: String, accept: (String, Boolean, String) -> Boolean): List<Span> {
    val spans = ArrayList<Span>()
    val buf = StringBuilder()
    var i = 0
    val n = text.length
    while (i < n) {
      var matched = false
      for (d in DELIMS) {
        if (!text.startsWith(d.left, i) || isEscaped(text, i)) continue
        var j = i + d.left.length
        var end = -1
        while (j < n) {
          val k = text.indexOf(d.right, j)
          if (k < 0) break
          if (isEscaped(text, k)) {
            j = k + 1
            continue
          }
          end = k
          break
        }
        if (end < 0) continue
        val inner = text.substring(i + d.left.length, end)
        if (inner.isBlank()) continue
        if (!d.newline && '\n' in inner) continue
        if (!accept(inner, d.display, d.left)) {
          buf.append(d.left)
          i += d.left.length
          matched = true
          break
        }
        if (buf.isNotEmpty()) {
          spans.add(Span(false, false, buf.toString()))
          buf.setLength(0)
        }
        spans.add(Span(true, d.display, inner))
        i = end + d.right.length
        matched = true
        break
      }
      if (!matched) {
        buf.append(text[i])
        i++
      }
    }
    if (buf.isNotEmpty()) spans.add(Span(false, false, buf.toString()))
    return spans
  }

  private fun acceptSingleDollar(inner: String): Boolean {
    val body = inner.trim()
    if (body.isEmpty()) return false
    if (STRONG_MATH.containsMatchIn(body)) return true
    if (body[0].isDigit() && PROSE_WORD.containsMatchIn(body)) return false
    return true
  }

  private fun placeholder(i: Int) = "\u0000MATH_${i}\u0000"

  private fun protectMath(text: String, formulaMode: Boolean): Pair<String, List<Pair<String, Boolean>>> {
    val store = ArrayList<Pair<String, Boolean>>()
    val out = StringBuilder()
    val spans =
      splitMath(text) { inner, _, left ->
        if (!formulaMode && HTML_BOUNDARY_IN_MATH.containsMatchIn(inner)) {
          false
        } else {
          !(left == "$" && !acceptSingleDollar(inner))
        }
      }
    for (s in spans) {
      if (!s.math) {
        out.append(s.payload)
        continue
      }
      out.append(placeholder(store.size))
      store.add(unescapeHtml(s.payload.trim()).replace('\u00a0', ' ') to s.display)
    }
    return out.toString() to store
  }

  private fun restoreMath(html: String, store: List<Pair<String, Boolean>>): String {
    var out = html
    for ((i, entry) in store.withIndex()) {
      val (latex, display) = entry
      val safe = latex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      val tag =
        if (display) {
          "<span class=\"math-display\">$$$safe$$</span>"
        } else {
          "<span class=\"math-inline\">\\($safe\\)</span>"
        }
      out = out.replace(placeholder(i), tag)
    }
    return out
  }

  // ------------------------------------------------------------ content

  /** The HTML fragment of [text] for [category] (render_label.prepare_render_content). */
  fun content(text: String, category: Category): String {
    if (text.isEmpty()) return ""
    if (category == Category.FORMULA) return markdownToHtml(prepareFormula(text), formulaMode = true)
    if (!IS_HTML.containsMatchIn(text)) return markdownToHtml(text, formulaMode = false)
    if (MARKDOWN_SYNTAX.containsMatchIn(text)) return markdownToHtml(text, formulaMode = false)
    var html = cleanHtmlStyle(text)
    html = html.replace("\n", "<br>")
    html = TABLE_TAG_THEN_BR.replace(html) { it.groupValues[1] }
    html = BR_THEN_TABLE_TAG.replace(html) { it.groupValues[1] }
    val (protected, store) = protectMath(html, formulaMode = false)
    return if (store.isEmpty()) protected else restoreMath(protected, store)
  }

  private val IS_HTML = Regex("<(?:table|div|p|h[1-6]|ul|ol|tr|td|th)\\b", RegexOption.IGNORE_CASE)
  private val MARKDOWN_SYNTAX =
    Regex(
      "(?m)^#{1,6}[ \\t]|^[ \\t]*[-*+][ \\t]+\\S|\\*\\*[^*\\n]+\\*\\*|^[ \\t]*[0-9]+\\.[ \\t]+\\S"
    )
  private val TABLE_TAG_THEN_BR =
    Regex("(</?(?:table|thead|tbody|tfoot|tr|th|td)[^>]*>)\\s*(?:<br>)+\\s*", RegexOption.IGNORE_CASE)
  private val BR_THEN_TABLE_TAG =
    Regex("(?:<br>)+\\s*(</?(?:table|thead|tbody|tfoot|tr|th|td)[^>]*>)", RegexOption.IGNORE_CASE)

  private fun cleanHtmlStyle(input: String): String {
    var html = input
    val ic = RegexOption.IGNORE_CASE
    html = Regex("margin:\\s*auto;?", ic).replace(html, "")
    html = Regex("width:\\s*max-content;?", ic).replace(html, "")
    html = Regex("text-align:\\s*center;?", ic).replace(html, "")
    html = Regex("(?:min-)?height:\\s*\\d+[^;\"']*;?", ic).replace(html, "")
    html =
      Regex(
          "(?:padding|margin)(?:-(?:top|bottom|left|right))?:\\s*(?:[5-9]\\d{1,}|[1-9]\\d{2,})px[^;\"']*;?",
          ic,
        )
        .replace(html, "")
    return html
  }

  private fun hasAnyMathDelim(text: String): Boolean {
    var i = 0
    while (i < text.length) {
      if (text[i] == '$' && !isEscaped(text, i)) return true
      if (i + 2 <= text.length) {
        val two = text.substring(i, i + 2)
        if ((two == "\\[" || two == "\\]" || two == "\\(" || two == "\\)") && !isEscaped(text, i)) {
          return true
        }
      }
      i++
    }
    return false
  }

  /** A bare formula body (no delimiter at all) becomes one display block. */
  fun prepareFormula(text: String): String {
    val t = text.trim()
    if (t.isEmpty()) return t
    return if (!hasAnyMathDelim(t)) "$$\n$t\n$$" else t
  }

  private fun markdownToHtml(input: String, formulaMode: Boolean): String {
    if (input.isEmpty()) return ""
    val (protected, store) = protectMath(input, formulaMode)
    // "3. xxx" in OCR text is content, not a list to renumber; ___ rules stay literal
    var text = Regex("(?m)^(\\s*\\d+)\\.\\s").replace(protected) { it.groupValues[1] + "\\. " }
    text = Regex("_{3,}").replace(text) { "\\_".repeat(it.value.length) }
    val html = MiniMarkdown.toHtml(text)
    return if (store.isEmpty()) html else restoreMath(html, store)
  }

  private fun unescapeHtml(s: String): String {
    if ('&' !in s) return s
    return Regex("&(#[0-9]+|#[xX][0-9a-fA-F]+|amp|lt|gt|quot|apos|nbsp);").replace(s) { m ->
      when (val e = m.groupValues[1]) {
        "amp" -> "&"
        "lt" -> "<"
        "gt" -> ">"
        "quot" -> "\""
        "apos" -> "'"
        "nbsp" -> "\u00a0"
        else ->
          if (e.startsWith("#x") || e.startsWith("#X")) {
            String(Character.toChars(e.substring(2).toInt(16)))
          } else {
            String(Character.toChars(e.substring(1).toInt()))
          }
      }
    }
  }

  // ------------------------------------------------------------ page

  /**
   * The page of render_label.wrap_html. Two changes for a phone: KaTeX 0.16.47 and DejaVu Sans come
   * from the app's assets (the source injects the same KaTeX files; its Linux reference deployment
   * draws Latin text in DejaVu Sans, the system's default sans-serif face there), so the phone's
   * own font setting does not change the picture; and KaTeX's auto-render, which never runs in the
   * source (it is not loaded when the page fires DOMContentLoaded), is left out.
   */
  fun page(content: String): String =
    """<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=$VIEWPORT_WIDTH, initial-scale=1, minimum-scale=1, maximum-scale=1, user-scalable=no">
<link rel="stylesheet" href="katex/katex.min.css">
<script src="katex/katex.min.js"></script>
<style>
@font-face { font-family: "DejaVu Sans"; src: url("fonts/DejaVuSans.ttf") format("truetype"); font-weight: normal; font-style: normal; }
@font-face { font-family: "DejaVu Sans"; src: url("fonts/DejaVuSans-Bold.ttf") format("truetype"); font-weight: bold; font-style: normal; }
body { font-family: "DejaVu Sans", -apple-system, BlinkMacSystemFont, "Segoe UI",
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
    "Microsoft YaHei", SimSun, sans-serif; padding: 0; margin: 0; background: #fff;
    font-size: 18px; line-height: 2.0; }
h1 { font-size: 1.8em; margin: 0.8em 0 0.4em; }
h2 { font-size: 1.5em; margin: 0.7em 0 0.4em; }
h3 { font-size: 1.3em; margin: 0.5em 0 0.3em; }
h4, h5, h6 { font-size: 1.1em; margin: 0.4em 0 0.3em; }
p { margin: 0.2em 0; }
table { border-collapse: collapse; margin: 0; width: 100%; max-width: 100%; table-layout: fixed; }
th, td { border: 1px solid #999; box-sizing: border-box; padding: 12px; word-break: break-word; overflow-wrap: anywhere; text-align: left; }
td, th { vertical-align: middle; }
table td, table th { white-space: normal; }
th { background-color: #f2f2f2; }
pre { background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }
code { background: #f0f0f0; padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #ddd; margin: 0.6em 0; padding: 0.4em 1em; color: #555; }
ul, ol { margin: 0.5em 0; padding-left: 1.5em; }
li { margin: 0.3em 0; }
.math-display { display: block; text-align: center; margin: 0.8em 0; }
#capture-area { display: inline-block; }
</style></head><body><div id="capture-area">$content</div>
<script>$SCRIPTS</script></body></html>"""

  /** The page of render_label._katex_error_page_html: a picture of KaTeX's error messages. */
  fun errorPage(errors: List<String>): String {
    val items = errors.joinToString("") { "<li><pre class=\"err-msg\">" + escapeHtml(it) + "</pre></li>" }
    return """<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=$VIEWPORT_WIDTH, initial-scale=1, minimum-scale=1, maximum-scale=1, user-scalable=no">
<style>
@font-face { font-family: "DejaVu Sans"; src: url("fonts/DejaVuSans.ttf") format("truetype"); }
body { margin: 0; padding: 0; background: #fff; font-family: "DejaVu Sans", -apple-system, "Segoe UI",
    "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
#capture-area { display: inline-block; max-width: 1100px; padding: 14px 18px;
    box-sizing: border-box; }
ol { margin: 0; padding-left: 1.4em; }
li { margin: 6px 0; }
.err-msg { color: #cc0000; font-family: Menlo, Consolas, monospace; font-size: 15px;
    white-space: pre-wrap; word-break: break-word; margin: 0; }
</style></head><body><div id="capture-area">
<ol>$items</ol>
</div>
<script>$ERROR_SCRIPT</script></body></html>"""
  }

  fun escapeHtml(s: String): String =
    s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&#x27;")

  /**
   * The source's in-page steps, in its order: render each protected math span with katex.render
   * (throwOnError, the message shown on an error), load every KaTeX face, wait for the fonts, the
   * table and long-text layout (TABLE_LAYOUT_SCRIPT, given the region's width), fit wide display
   * formulas, wait 300 ms; then report the capture area's box and any KaTeX error to the app.
   */
  private val SCRIPTS =
    """
function renderMath() {
  if (typeof katex === 'undefined') return false;
  const els = document.querySelectorAll('#capture-area .math-display, #capture-area .math-inline');
  for (const el of els) {
    if (el.dataset.mathDone) continue;
    el.dataset.mathDone = '1';
    let body = el.textContent;
    const disp = el.classList.contains('math-display');
    if (disp && body.startsWith('$$') && body.endsWith('$$')) {
      body = body.slice(2, -2);
    } else if (!disp && body.startsWith('\\(') && body.endsWith('\\)')) {
      body = body.slice(2, -2);
    }
    try {
      katex.render(body, el, {throwOnError: true, displayMode: disp});
    } catch (e) {
      const span = document.createElement('span');
      span.className = 'katex-error';
      span.style.color = '#cc0000';
      span.title = String(e);
      span.textContent = String(e);
      el.replaceChildren(span);
    }
  }
  return true;
}
async function preloadKatexFonts() {
  if (typeof katex === 'undefined') return;
  const fams = ['KaTeX_Main','KaTeX_Math','KaTeX_AMS','KaTeX_Size1','KaTeX_Size2',
                'KaTeX_Size3','KaTeX_Size4','KaTeX_Caligraphic','KaTeX_Fraktur',
                'KaTeX_Script','KaTeX_SansSerif','KaTeX_Typewriter'];
  const jobs = [];
  for (const f of fams) for (const s of ['normal','italic']) for (const w of ['normal','bold'])
    jobs.push(document.fonts.load(s + ' ' + w + ' 16px "' + f + '"').catch(() => {}));
  await Promise.all(jobs);
}
const TABLE_LAYOUT = $TABLE_LAYOUT_SCRIPT;
function fitDisplayMath() {
  const area = document.getElementById('capture-area');
  if (!area) return;
  for (const d of area.querySelectorAll('.katex-display')) {
    const avail = d.clientWidth || area.clientWidth;
    const need = d.scrollWidth;
    if (need > avail + 1) {
      const scale = Math.max(0.3, avail / need);
      d.style.fontSize = (scale * 100).toFixed(1) + '%';
    }
  }
}
function report() {
  const errors = Array.from(document.querySelectorAll('.katex-error')).map((el) => {
    const title = (el.getAttribute('title') || '').trim();
    if (title) return title;
    return 'KaTeX error: ' + (el.textContent || '').slice(0, 300);
  });
  const r = document.getElementById('capture-area').getBoundingClientRect();
  return JSON.stringify({errors: errors, x: r.left + window.scrollX, y: r.top + window.scrollY,
                         width: r.width, height: r.height, dpr: window.devicePixelRatio,
                         inner: window.innerWidth});
}
async function renderAll(targetWidth) {
  try {
    renderMath();
    await preloadKatexFonts();
    await document.fonts.ready;
    TABLE_LAYOUT(targetWidth);
    renderMath();
    await document.fonts.ready;
    fitDisplayMath();
    await new Promise((r) => setTimeout(r, 300));
    OcrBridge.done(report());
  } catch (e) {
    OcrBridge.failed(String(e));
  }
}
"""

  private const val ERROR_SCRIPT =
    """
async function renderAll(targetWidth) {
  await document.fonts.ready;
  const r = document.getElementById('capture-area').getBoundingClientRect();
  OcrBridge.done(JSON.stringify({errors: [], x: r.left, y: r.top, width: r.width, height: r.height,
                                 dpr: window.devicePixelRatio, inner: window.innerWidth}));
}
"""

  /** render_label.TABLE_LAYOUT_SCRIPT, unchanged. */
  private const val TABLE_LAYOUT_SCRIPT =
    """(targetWidth) => {
    const area = document.getElementById('capture-area');
    if (!area) return;
    const tables = Array.from(area.querySelectorAll('table'));

    const visualLength = (text) => {
        let score = 0;
        for (const ch of Array.from(text || '')) {
            score += /[\u2e80-\u9fff\uff00-\uffef]/.test(ch) ? 2 : 1;
        }
        return score;
    };

    if (!tables.length) {
        const MIN_TEXT_WIDTH = 480;
        const MAX_TEXT_WIDTH = 1200;
        const MAX_ASPECT_RATIO = 8;
        const MIN_MEANINGFUL_WIDTH = 1;

        const clearTextWidth = () => {
            area.classList.remove('long-text-wrap');
            area.style.width = '';
            area.style.maxWidth = '';
            area.style.whiteSpace = '';
            area.style.wordBreak = '';
            area.style.overflowWrap = '';
            area.style.boxSizing = '';
            for (const el of area.querySelectorAll('*')) {
                el.style.maxWidth = '';
                el.style.wordBreak = '';
                el.style.overflowWrap = '';
            }
        };

        const applyTextWidth = (width) => {
            const px = `${'$'}{Math.round(width)}px`;
            area.classList.add('long-text-wrap');
            area.style.width = px;
            area.style.maxWidth = px;
            area.style.whiteSpace = 'normal';
            area.style.wordBreak = 'break-word';
            area.style.overflowWrap = 'anywhere';
            area.style.boxSizing = 'border-box';
            for (const el of area.querySelectorAll('*')) {
                const katexRoot = el.closest('.katex');
                if (katexRoot) {
                    if (el === katexRoot) { el.style.maxWidth = '100%'; }
                    continue;
                }
                el.style.maxWidth = '100%';
                if (!['PRE', 'CODE'].includes(el.tagName)) {
                    el.style.wordBreak = 'break-word';
                    el.style.overflowWrap = 'anywhere';
                }
            }
        };

        const taggedDisplays = Array.from(area.querySelectorAll('.katex-display')).filter(
            (display) => display.querySelector('.katex-html > .tag')
        );
        if (taggedDisplays.length) {
            clearTextWidth();
            let requiredWidth = 0;
            for (const display of taggedDisplays) {
                const katexHtml = display.querySelector('.katex-html');
                const tag = katexHtml?.querySelector(':scope > .tag');
                if (!katexHtml || !tag) continue;
                const bodyWidth = Array.from(katexHtml.children)
                    .filter((child) => child !== tag)
                    .reduce((sum, child) => sum + child.getBoundingClientRect().width, 0);
                const tagWidth = tag.getBoundingClientRect().width;
                requiredWidth = Math.max(requiredWidth, bodyWidth + 2 * tagWidth + 32);
            }

            const requestedWidth = Number(targetWidth);
            const boundedTarget = Number.isFinite(requestedWidth) && requestedWidth > 0
                ? Math.min(requestedWidth, MAX_TEXT_WIDTH)
                : 0;
            const width = Math.max(requiredWidth, boundedTarget);
            if (width > 0) {
                const px = `${'$'}{Math.ceil(width)}px`;
                area.style.width = px;
                area.style.maxWidth = px;
                area.style.boxSizing = 'border-box';
            }
            return;
        }

        const measure = () => {
            const rect = area.getBoundingClientRect();
            const width = Math.max(MIN_MEANINGFUL_WIDTH, rect.width || area.scrollWidth || 0);
            const height = Math.max(MIN_MEANINGFUL_WIDTH, rect.height || area.scrollHeight || 0);
            return {width, height, ratio: width / height};
        };

        clearTextWidth();
        const natural = measure();
        if (natural.ratio <= MAX_ASPECT_RATIO || natural.width <= MIN_TEXT_WIDTH) return;

        let high = Math.min(MAX_TEXT_WIDTH, natural.width);
        applyTextWidth(high);
        const highMeasure = measure();
        if (highMeasure.ratio <= MAX_ASPECT_RATIO) return;

        let low = MIN_TEXT_WIDTH;
        applyTextWidth(low);
        const lowMeasure = measure();
        if (lowMeasure.ratio > MAX_ASPECT_RATIO) return;

        let best = low;
        for (let i = 0; i < 12; i += 1) {
            const mid = (low + high) / 2;
            applyTextWidth(mid);
            const current = measure();
            if (current.ratio <= MAX_ASPECT_RATIO) {
                best = mid;
                low = mid;
            } else {
                high = mid;
            }
        }
        applyTextWidth(best);
        return;
    }

    const maxWidth = Math.max(900, Math.min(Number(targetWidth) || 1200, 1200));
    area.style.width = `${'$'}{maxWidth}px`;
    area.style.maxWidth = `${'$'}{maxWidth}px`;

    const cellScore = (cell) => {
        const text = (cell.innerText || '').replace(/\s+/g, ' ').trim();
        if (!text) return 1;
        const tokens = text.split(/\s+/).filter(Boolean);
        const longestToken = tokens.reduce((m, token) => Math.max(m, visualLength(token)), 0);
        const total = visualLength(text);
        return Math.max(4, Math.min(80, Math.max(longestToken, Math.sqrt(total) * 6)));
    };

    for (const table of tables) {
        table.querySelectorAll('colgroup').forEach((el) => el.remove());
        table.removeAttribute('width');
        table.style.width = '100%';
        table.style.maxWidth = '100%';
        table.style.minWidth = '0';
        table.style.tableLayout = 'fixed';
        for (const cell of table.querySelectorAll('th, td')) {
            cell.removeAttribute('width');
            cell.style.width = '';
            cell.style.minWidth = '0';
            cell.style.maxWidth = '';
            cell.style.whiteSpace = 'normal';
            cell.style.wordBreak = 'break-word';
            cell.style.overflowWrap = 'anywhere';
        }

        const rows = Array.from(table.rows);
        const colCount = rows.reduce((maxCols, row) => {
            const cols = Array.from(row.cells).reduce((sum, cell) => sum + Math.max(1, cell.colSpan || 1), 0);
            return Math.max(maxCols, cols);
        }, 0);
        if (!colCount) continue;

        const weights = Array(colCount).fill(1);
        const pending = Array(colCount).fill(0);
        for (const row of rows) {
            let col = 0;
            for (const cell of row.cells) {
                while (col < colCount && pending[col] > 0) col += 1;
                const span = Math.max(1, cell.colSpan || 1);
                const rspan = Math.max(1, cell.rowSpan || 1);
                const score = cellScore(cell) / span;
                for (let i = 0; i < span && col + i < colCount; i += 1) {
                    weights[col + i] = Math.max(weights[col + i], score);
                    if (rspan > 1) pending[col + i] = Math.max(pending[col + i], rspan);
                }
                col += span;
            }
            for (let i = 0; i < colCount; i += 1) if (pending[i] > 0) pending[i] -= 1;
        }

        const totalWeight = weights.reduce((sum, value) => sum + value, 0) || colCount;
        const colgroup = document.createElement('colgroup');
        colgroup.dataset.autoLayout = 'true';
        for (const weight of weights) {
            const col = document.createElement('col');
            col.style.width = `${'$'}{(weight / totalWeight) * 100}%`;
            colgroup.appendChild(col);
        }
        table.prepend(colgroup);
    }
}"""
}

/**
 * The part of Python-Markdown 3.10.3 (with nl2br) that OCR text of a region uses: its whitespace
 * normalisation, paragraphs, line breaks, headings, bullet lists, backslash escapes, code spans,
 * links, inline HTML, emphasis (asterisks inside words, underscores only at word edges) and the
 * HTML escapes. The math placeholders (NUL-delimited) pass through unchanged.
 */
object MiniMarkdown {
  private const val ESCAPABLE = "\\`*_{}[]()>#+-.!"

  fun toHtml(source: String): String {
    var text = source.replace("\r\n", "\n").replace('\r', '\n')
    text = expandTabs(text, 4)
    text = text.split('\n').joinToString("\n") { if (it.isBlank()) "" else it }
    val out = ArrayList<String>()
    val lines = text.split('\n')
    var i = 0
    val para = ArrayList<String>()
    fun flushPara() {
      if (para.isNotEmpty()) {
        out.add("<p>" + inline(para.joinToString("\n").trimStart()) + "</p>")
        para.clear()
      }
    }
    while (i < lines.size) {
      val line = lines[i]
      when {
        line.isEmpty() -> flushPara()
        HEADING.matches(line) && para.isEmpty() -> {
          val m = HEADING.matchEntire(line)!!
          val level = m.groupValues[1].length
          val body = m.groupValues[2].trim().trimEnd('#').trim()
          out.add("<h$level>" + inline(body) + "</h$level>")
        }
        BULLET.containsMatchIn(line) && para.isEmpty() -> {
          val items = ArrayList<String>()
          while (i < lines.size && lines[i].isNotEmpty()) {
            val m = BULLET.find(lines[i])
            if (m != null) {
              items.add(lines[i].substring(m.range.last + 1))
            } else if (items.isNotEmpty()) {
              items[items.size - 1] = items.last() + "\n" + lines[i].trim()
            }
            i++
          }
          out.add("<ul>\n" + items.joinToString("\n") { "<li>" + inline(it) + "</li>" } + "\n</ul>")
          continue
        }
        else -> para.add(line)
      }
      i++
    }
    flushPara()
    return out.joinToString("\n")
  }

  private val HEADING = Regex("^(#{1,6})(.*)$")
  private val BULLET = Regex("^[ ]{0,3}[*+-][ ]+")

  private fun expandTabs(s: String, size: Int): String {
    if ('\t' !in s) return s
    val sb = StringBuilder()
    var col = 0
    for (c in s) {
      when (c) {
        '\t' -> {
          val n = size - col % size
          repeat(n) { sb.append(' ') }
          col += n
        }
        '\n' -> {
          sb.append(c)
          col = 0
        }
        else -> {
          sb.append(c)
          col++
        }
      }
    }
    return sb.toString()
  }

  private val RAW_TAG = Regex("<(/?[a-zA-Z][^<>@ ]*( [^<>]*)?|!--(?:(?!<!--|-->).)*--)>")
  private val ENTITY = Regex("&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z0-9]+);")
  private val LINK = Regex("\\[([^\\[\\]]*)\\]\\(\\s*<?([^\\s)>]*)>?\\s*\\)")

  private fun isWord(c: Char) = c.isLetterOrDigit() || c == '_'

  fun inline(s: String): String {
    val sb = StringBuilder()
    var i = 0
    while (i < s.length) {
      val c = s[i]
      when {
        c == '\\' && i + 1 < s.length && s[i + 1] in ESCAPABLE -> {
          sb.append(escape(s[i + 1]))
          i += 2
        }
        c == '`' -> {
          var run = 0
          while (i + run < s.length && s[i + run] == '`') run++
          val close = s.indexOf("`".repeat(run), i + run)
          if (close >= 0) {
            sb.append("<code>").append(escapeAll(s.substring(i + run, close).trim())).append("</code>")
            i = close + run
          } else {
            sb.append("`".repeat(run))
            i += run
          }
        }
        c == '<' && RAW_TAG.matchesAt(s, i) -> {
          val m = RAW_TAG.matchAt(s, i)!!
          sb.append(m.value)
          i += m.value.length
        }
        c == '&' && ENTITY.matchesAt(s, i) -> {
          val m = ENTITY.matchAt(s, i)!!
          sb.append(m.value)
          i += m.value.length
        }
        c == '[' && LINK.matchesAt(s, i) -> {
          val m = LINK.matchAt(s, i)!!
          sb.append("<a href=\"").append(m.groupValues[2].replace("&", "&amp;").replace("\"", "&quot;"))
            .append("\">").append(inline(m.groupValues[1])).append("</a>")
          i += m.value.length
        }
        c == '*' -> i = emphasis(s, i, '*', sb)
        c == '_' -> i = emphasis(s, i, '_', sb)
        c == '\n' -> {
          sb.append("<br />\n")
          i++
        }
        else -> {
          sb.append(escape(c))
          i++
        }
      }
    }
    return sb.toString()
  }

  /** `**x**` / `__x__` strong, `*x*` / `_x_` emphasis; underscores only at word edges. */
  private fun emphasis(s: String, i: Int, mark: Char, sb: StringBuilder): Int {
    val before = if (i > 0) s[i - 1] else ' '
    var run = 0
    while (i + run < s.length && s[i + run] == mark && run < 3) run++
    val after = if (i + run < s.length) s[i + run] else ' '
    // a lone run between spaces stays literal (Python-Markdown's not_strong)
    if ((i == 0 || before.isWhitespace()) && after.isWhitespace()) {
      sb.append(mark.toString().repeat(run))
      return i + run
    }
    if (mark == '_' && isWord(before)) {
      sb.append(mark.toString().repeat(run))
      return i + run
    }
    for (n in minOf(run, 2) downTo 1) {
      val open = mark.toString().repeat(n)
      var k = s.indexOf(open, i + n)
      while (k >= 0) {
        val inner = s.substring(i + n, k)
        val nextOk = mark != '_' || k + n >= s.length || !isWord(s[k + n])
        val notMore = mark != '_' || k == 0 || s[k - 1] != '_'
        if (inner.isNotEmpty() && !inner[0].isWhitespace() && (n == 2 || mark != '*' || '*' !in inner) &&
          nextOk && notMore) {
          val tag = if (n == 2) "strong" else "em"
          sb.append("<$tag>").append(inline(inner)).append("</$tag>")
          return k + n
        }
        k = s.indexOf(open, k + 1)
      }
    }
    sb.append(mark)
    return i + 1
  }

  private fun escape(c: Char): String =
    when (c) {
      '&' -> "&amp;"
      '<' -> "&lt;"
      '>' -> "&gt;"
      else -> c.toString()
    }

  private fun escapeAll(s: String): String = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
}
