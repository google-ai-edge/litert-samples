/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.aiedge.examples.reinforcement_learning

import android.content.Context
import android.graphics.Point
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val helper: ReinforcementLearningHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val helper = ReinforcementLearningHelper(context)
                return MainViewModel(helper) as T
            }
        }
    }

    private val agentBoard = MutableSharedFlow<List<List<Int>>>()
    private val agentHit = MutableStateFlow(0)
    private val playerBoard = MutableSharedFlow<List<List<Int>>>()
    private val playerHit = MutableStateFlow(0)

    val uiState: StateFlow<UiState> = combine(
        playerBoard,
        agentBoard,
        agentHit,
        playerHit,
    ) { playerBoard, agentBoard, agentHit, playerHit ->
        UiState(
            displayAgentBoard = agentBoard,
            displayPlayerBoard = playerBoard,
            hiddenPlayerBoard = playerBoard.map {
                it.map { cell ->
                    if (cell == Cell.PLANE) Cell.EMPTY else cell
                }
            },
            agentHits = agentHit,
            playerHits = playerHit
        )
    }.stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState()
    )

    fun hitAgent(hitPoint: Point) {
        viewModelScope.launch {
            val displayAgentBoard = uiState.value.displayAgentBoard
            val newAgentBoard = displayAgentBoard.map { it.toMutableList() }

            val row = hitPoint.x
            val col = hitPoint.y
            if (newAgentBoard[row][col] == Cell.PLANE) {
                newAgentBoard[row][col] = Cell.HIT
                agentHit.update { it + 1 }
            } else {
                newAgentBoard[row][col] = Cell.MISS
            }
            // Player attack
            agentBoard.emit(newAgentBoard)

            // Agent attack if game is not over yet
            if (agentHit.value <= 8) {
                predict()
            }

        }
    }

    private fun agentHit(position: Int): Point {
        val row = position / uiState.value.displayPlayerBoard.size
        val col = position % uiState.value.displayPlayerBoard.size
        return Point(row, col)
    }

    private fun predict() {
        viewModelScope.launch {
            val displayPlayerBoard = uiState.value.displayPlayerBoard.map {
                it.toMutableList()
            }

            val hiddenPlayerBoard = uiState.value.hiddenPlayerBoard
            val hitPosition = helper.predict(hiddenPlayerBoard)
            val agentHit = agentHit(hitPosition)
            val row = agentHit.x
            val col = agentHit.y
            if (displayPlayerBoard[row][col] == Cell.PLANE) {
                displayPlayerBoard[row][col] = Cell.HIT
                playerHit.update { it + 1 }
            } else {
                displayPlayerBoard[row][col] = Cell.MISS
            }

            playerBoard.emit(displayPlayerBoard)
        }
    }

    fun reset() {
        viewModelScope.launch {
            agentBoard.emit(placeShip(board = EMPTY_BOARD))
            playerBoard.emit(placeShip(board = EMPTY_BOARD))
            agentHit.emit(0)
            playerHit.emit(0)
        }
    }
}
