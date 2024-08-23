/*
 * Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

package com.google.edgeai.examples.reinforcement_learning

import android.graphics.Point
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.painter.ColorPainter
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

        setContent {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            Scaffold(
                topBar = {
                    Header()
                },
                content = { paddingValue ->
                    Column(
                        modifier = Modifier.padding(
                            top = paddingValue.calculateTopPadding(),
                            start = 10.dp,
                            end = 10.dp
                        )
                    ) {
                        val agentBoard = uiState.displayAgentBoard
                        val playerBoard = uiState.displayPlayerBoard
                        val theEnd = uiState.theEnd

                        Text(
                            modifier = Modifier.align(Alignment.CenterHorizontally),
                            text = "Tap any cell in the agent board as your strike.\nRed-hit, Yellow-miss"
                        )
                        Spacer(modifier = Modifier.height(5.dp))
                        Row {
                            Board(user = User.Agent, board = agentBoard) { row, col ->
                                if (theEnd) return@Board
                                viewModel.hitAgent(
                                    hitPoint = Point(row, col)
                                )
                            }
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(text = "Agent board:\n${uiState.agentHits} hits")
                        }

                        Divider(modifier = Modifier.padding(vertical = 5.dp))

                        Row {
                            Board(user = User.Player, board = playerBoard) { _, _ -> }
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(text = "Your board:\n${uiState.playerHits} hits")
                        }
                        if (theEnd) {
                            Spacer(modifier = Modifier.height(10.dp))
                            val youWin = uiState.agentHits == 8

                            Text(
                                modifier = Modifier.align(Alignment.CenterHorizontally),
                                text = if (youWin) "You win!!!" else "Agent win!!!"
                            )
                            Button(
                                modifier = Modifier.align(Alignment.CenterHorizontally),
                                onClick = { viewModel.reset() }) {
                                Text(text = "Reset")
                            }
                        }
                    }
                },
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Header(modifier: Modifier = Modifier) {
    TopAppBar(
        modifier = modifier.height(40.dp),
        colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.LightGray),
        title = {
            Row(
                modifier = Modifier.fillMaxSize(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Image(
                    modifier = Modifier.size(30.dp),
                    painter = ColorPainter(color = Color.White),
                    contentDescription = null,
                )

                Spacer(modifier = modifier.width(10.dp))
                Text(text = "LiteRT", color = Color.Blue, fontWeight = FontWeight.SemiBold)
            }
        },
    )
}


@Composable
fun Board(
    modifier: Modifier = Modifier,
    board: List<List<Int>>,
    user: User,
    onHit: (row: Int, col: Int) -> Unit
) {
    Column(modifier = modifier) {
        for (row in board.indices) {
            Row {
                for (col in board[row].indices) {
                    val cellValue = board[row][col]
                    val color = when (cellValue) {
                        Cell.EMPTY -> Color.White
                        Cell.HIT -> Color.Red
                        Cell.PLANE -> if (user == User.Agent) Color.White else Color.Blue
                        Cell.MISS -> Color.Yellow
                        else -> Color.White
                    }
                    Box(
                        modifier = Modifier
                            .size(28.dp)
                            .background(color = color)
                            .border(1.dp, color = Color.Gray)
                            .clickable {
                                if (cellValue == Cell.EMPTY || cellValue == Cell.PLANE) {
                                    onHit(row, col)
                                }
                            },
                    )
                }
            }
        }
    }
}

enum class User {
    Player, Agent
}
