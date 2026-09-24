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

package com.google.ai.edge.examples.model_zoo.view

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.model_zoo.MainViewModel
import com.google.ai.edge.examples.model_zoo.UiState
import com.google.ai.edge.examples.model_zoo.data.DownloadState
import com.google.ai.edge.examples.model_zoo.data.DownloadStatus
import com.google.ai.edge.examples.model_zoo.data.ModelEntry
import java.util.Locale

/** Two compact cards per row; empty catalog groups do not occupy space. */
@Composable
internal fun HomeScreen(state: UiState, vm: MainViewModel) {
  val groups =
    listOf("Vision", "Audio")
      .map { group -> group to state.tasks.filter { it.group == group } }
      .filter { (_, entries) -> entries.isNotEmpty() }
  LazyVerticalGrid(
    columns = GridCells.Fixed(2),
    modifier = Modifier.fillMaxSize(),
    contentPadding = PaddingValues(16.dp),
    horizontalArrangement = Arrangement.spacedBy(8.dp),
    verticalArrangement = Arrangement.spacedBy(8.dp),
  ) {
    item(span = { GridItemSpan(maxLineSpan) }) {
      Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(
          "On-device vision & audio",
          style = MaterialTheme.typography.titleLarge.copy(fontSize = 18.sp, lineHeight = 24.sp),
        )
        val ready = state.tasks.count { state.downloads[it.taskId]?.status == DownloadStatus.READY }
        Text(
          "${state.tasks.size} tasks · $ready ready",
          style = MaterialTheme.typography.bodySmall,
          color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
      }
    }
    groups.forEach { (group, entries) ->
      item(key = "group-$group", span = { GridItemSpan(maxLineSpan) }) {
        Text(group, Modifier.padding(top = 4.dp), style = MaterialTheme.typography.titleSmall)
      }
      items(entries, key = { it.taskId }) { entry ->
        ExploreTaskCard(entry, state.downloads[entry.taskId]) { vm.navigate("task", entry.taskId) }
      }
    }
  }
}

@Composable
private fun ExploreTaskCard(entry: ModelEntry, download: DownloadState?, onClick: () -> Unit) {
  val ready = download?.status == DownloadStatus.READY
  val status =
    when (download?.status ?: DownloadStatus.MISSING) {
      DownloadStatus.MISSING -> "Not downloaded"
      DownloadStatus.STARTING -> "Starting…"
      DownloadStatus.DOWNLOADING -> {
        val percent =
          if ((download?.totalBytes ?: 0L) > 0L)
            ((download!!.receivedBytes.toDouble() / download.totalBytes) * 100)
              .toInt()
              .coerceIn(0, 100)
          else 0
        "$percent% downloading"
      }
      DownloadStatus.PAUSED -> "Paused"
      DownloadStatus.VERIFYING -> "Checking…"
      DownloadStatus.READY -> "Ready"
      DownloadStatus.ERROR -> "Download failed"
    }
  val size = compactSize(entry.totalBytes)
  Card(
    modifier =
      Modifier.fillMaxWidth().height(100.dp).clickable(onClick = onClick).semantics(
        mergeDescendants = true
      ) {
        contentDescription = "${entry.task}, ${entry.model}, $size, $status"
      },
    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
    shape = MaterialTheme.shapes.large,
  ) {
    Column(Modifier.fillMaxSize().padding(8.dp), verticalArrangement = Arrangement.SpaceBetween) {
      Row(
        Modifier.fillMaxWidth().height(34.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
      ) {
        Icon(
          taskIcon(entry.taskId),
          contentDescription = null,
          modifier = Modifier.size(22.dp),
          tint = MaterialTheme.colorScheme.primary,
        )
        Text(
          entry.task,
          Modifier.weight(1f),
          style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
          maxLines = 2,
          overflow = TextOverflow.Ellipsis,
        )
      }
      Text(
        entry.model,
        style = MaterialTheme.typography.bodySmall.copy(lineHeight = 14.sp),
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        maxLines = 2,
        overflow = TextOverflow.Ellipsis,
      )
      Surface(
        color =
          if (ready) MaterialTheme.colorScheme.primaryContainer
          else MaterialTheme.colorScheme.surfaceVariant,
        contentColor =
          if (ready) MaterialTheme.colorScheme.onPrimaryContainer
          else MaterialTheme.colorScheme.onSurfaceVariant,
        shape = MaterialTheme.shapes.small,
      ) {
        Text(
          "$size · $status",
          Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
          style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, lineHeight = 14.sp),
          maxLines = 1,
          overflow = TextOverflow.Ellipsis,
        )
      }
    }
  }
}

private fun compactSize(bytes: Long): String =
  when {
    bytes <= 0L -> "—"
    bytes >= 1_000_000_000L -> String.format(Locale.ENGLISH, "%.2f GB", bytes / 1_000_000_000.0)
    bytes >= 10_000_000L -> String.format(Locale.ENGLISH, "%.0f MB", bytes / 1_000_000.0)
    else -> String.format(Locale.ENGLISH, "%.1f MB", bytes / 1_000_000.0)
  }
