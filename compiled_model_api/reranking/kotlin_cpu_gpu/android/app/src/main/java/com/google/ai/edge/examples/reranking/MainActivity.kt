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

package com.google.ai.edge.examples.reranking

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
 * On-device RAG reranking with Qwen3-Reranker-0.6B on CompiledModel GPU.
 * Scores each bundled candidate document against a typed query by P("yes") relevance and ranks them.
 */
class MainActivity : Activity() {

    private lateinit var reranker: Reranker
    private lateinit var status: TextView
    private lateinit var results: TextView
    private lateinit var query: EditText
    private lateinit var search: Button
    private var docs: List<String> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(40, 72, 40, 40)
        }
        root.addView(TextView(this).apply {
            text = "RAG Reranker — Qwen3-Reranker-0.6B (GPU)"
            textSize = 18f; setTextColor(Color.BLACK); setPadding(0, 0, 0, 24)
        })
        query = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT
            hint = "Type a query…"; setText("What is the capital of China?")
        }
        root.addView(query)
        search = Button(this).apply { text = "Rerank"; isEnabled = false }
        root.addView(search)
        status = TextView(this).apply { setPadding(0, 16, 0, 16); setTextColor(Color.DKGRAY) }
        root.addView(status)
        results = TextView(this).apply { textSize = 15f; setTextColor(Color.BLACK) }
        root.addView(results)
        setContentView(ScrollView(this).apply {
            addView(root, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        })

        docs = assets.open("docs.txt").bufferedReader().readLines().filter { it.isNotBlank() }
        status.text = "Loading model (GPU compile ~40 s on first run)…"

        Thread {
            try {
                reranker = Reranker(this)
                runOnUiThread {
                    status.text = "Ready — ${docs.size} candidate documents. Enter a query."
                    search.isEnabled = true
                    search.setOnClickListener { runRerank() }
                }
                rerank(query.text.toString())          // auto-demo on first load
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Load failed: ${e.message}" }
            }
        }.start()
    }

    private fun runRerank() {
        search.isEnabled = false
        Thread { rerank(query.text.toString()) }.start()
    }

    private fun rerank(q: String) {
        if (q.isBlank()) return
        val t0 = System.currentTimeMillis()
        val ranked = docs.map { it to reranker.score(q, it) }.sortedByDescending { it.second }
        val dt = System.currentTimeMillis() - t0
        Log.i("RERANK", "QUERY(${dt}ms): $q")
        ranked.forEachIndexed { r, p -> Log.i("RERANK", String.format("  %2d. %.3f  %s", r + 1, p.second, p.first)) }
        runOnUiThread {
            status.text = "Reranked ${docs.size} docs in ${dt} ms"
            results.text = ranked.joinToString("\n") { String.format("%.3f  %s", it.second, it.first) }
            search.isEnabled = true
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::reranker.isInitialized) reranker.close()
    }
}
