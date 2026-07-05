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

package com.google.ai.edge.examples.semantic_similarity

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * On-device semantic search / RAG retrieval with Qwen3-Embedding-0.6B on CompiledModel GPU.
 * Embeds a bundled corpus once, then ranks it against a typed query by cosine similarity.
 */
class MainActivity : Activity() {

    private lateinit var embedder: TextEmbedder
    private lateinit var status: TextView
    private lateinit var results: TextView
    private lateinit var query: EditText
    private lateinit var search: Button

    private var corpus: List<String> = emptyList()
    private var corpusVecs: Array<FloatArray> = emptyArray()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 72, 40, 40)
        }
        root.addView(TextView(this).apply {
            text = "Semantic Search — Qwen3-Embedding-0.6B (GPU)"
            textSize = 18f; setTextColor(Color.BLACK); setPadding(0, 0, 0, 24)
        })
        query = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT
            hint = "Type a query…"; setText("What is the capital of China?")
        }
        root.addView(query)
        search = Button(this).apply { text = "Search"; isEnabled = false }
        root.addView(search)
        status = TextView(this).apply { setPadding(0, 16, 0, 16); setTextColor(Color.DKGRAY) }
        root.addView(status)
        results = TextView(this).apply { textSize = 15f; setTextColor(Color.BLACK) }
        root.addView(results)

        setContentView(ScrollView(this).apply {
            addView(root, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        })

        corpus = assets.open("corpus.txt").bufferedReader().readLines().filter { it.isNotBlank() }
        status.text = "Loading model + embedding ${corpus.size} documents (GPU compile ~40 s on first run)…"

        Thread {
            try {
                val t0 = System.currentTimeMillis()
                embedder = TextEmbedder(this)
                corpusVecs = Array(corpus.size) { embedder.embed(corpus[it]) }
                val dt = System.currentTimeMillis() - t0
                Log.i("TEAPP", "READY: ${corpus.size} docs embedded in ${dt} ms (incl. one-time GPU compile)")
                // auto demo search so the pipeline is verifiable headlessly (screen may be locked)
                val q0 = query.text.toString()
                val qt = System.currentTimeMillis()
                val qv = embedder.embedQuery(q0)
                val ranked = corpus.indices.map { it to cosine(qv, corpusVecs[it]) }
                    .sortedByDescending { it.second }
                Log.i("TEAPP", "QUERY(${System.currentTimeMillis() - qt}ms): $q0")
                ranked.forEachIndexed { r, p ->
                    Log.i("TEAPP", String.format("  %2d. %.3f  %s", r + 1, p.second, corpus[p.first]))
                }
                runOnUiThread {
                    status.text = "Ready — ${corpus.size} docs embedded in ${dt} ms. Enter a query."
                    results.text = ranked.joinToString("\n") {
                        String.format("%.3f  %s", it.second, corpus[it.first]) }
                    search.isEnabled = true
                    search.setOnClickListener { runSearch() }
                }
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Load failed: ${e.message}" }
            }
        }.start()
    }

    private fun runSearch() {
        val q = query.text.toString().ifBlank { return }
        search.isEnabled = false
        status.text = "Embedding query…"
        Thread {
            val t0 = System.currentTimeMillis()
            val qv = embedder.embedQuery(q)
            val ranked = corpus.indices
                .map { it to cosine(qv, corpusVecs[it]) }
                .sortedByDescending { it.second }
            val dt = System.currentTimeMillis() - t0
            val sb = StringBuilder()
            for ((rank, pair) in ranked.withIndex()) {
                val (i, score) = pair
                sb.append(String.format("%2d. %.3f  %s\n", rank + 1, score, corpus[i]))
            }
            runOnUiThread {
                status.text = "Query embedded + ranked in ${dt} ms"
                results.text = sb.toString()
                search.isEnabled = true
            }
        }.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::embedder.isInitialized) embedder.close()
    }
}
