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

import androidx.compose.runtime.Immutable
import kotlin.random.Random

@Immutable
data class UiState(
    val displayPlayerBoard: List<List<Int>> = placeShip(board = EMPTY_BOARD),
    val displayAgentBoard: List<List<Int>> = placeShip(board = EMPTY_BOARD),
    val hiddenPlayerBoard: List<List<Int>> = EMPTY_BOARD,
    val agentHits: Int = 0,
    val playerHits: Int = 0,
    val theEnd: Boolean = agentHits == 8 || playerHits == 8 // 8 is plane size
)

// Define a data class to represent a cell on the board
data class Cell(val value: Int) {
    companion object {
        const val EMPTY = 0
        const val PLANE = 2
        const val HIT = 1
        const val MISS = -1
    }
}

enum class Orientation {
    HeadingRight,
    HeadingUp,
    HeadingLeft,
    HeadingDown
}

// Create the empty board(8x8) using a 2D array
val EMPTY_BOARD = List(8) {
    List(8) { Cell.EMPTY }
}

// Function to place a plane on the board (consider error handling later)
fun placeShip(
    board: List<List<Int>>,
): List<List<Int>> {
    val resultBoard = board.map { it.toMutableList() }
    val orientation = Orientation.entries.random()
    val boardSize = resultBoard.size
    val planeCoreX: Int
    val planeCoreY: Int
    when (orientation) {
        Orientation.HeadingRight -> {
            planeCoreX = Random.nextInt(boardSize - 2) + 1
            planeCoreY = Random.nextInt(boardSize - 3) + 2
            resultBoard[planeCoreX][planeCoreY - 2] = Cell.PLANE
            resultBoard[planeCoreX][planeCoreY - 2] = Cell.PLANE
            resultBoard[planeCoreX - 1][planeCoreY - 2] = Cell.PLANE
            resultBoard[planeCoreX + 1][planeCoreY - 2] = Cell.PLANE

        }

        Orientation.HeadingUp -> {
            planeCoreX = Random.nextInt(boardSize - 3) + 1
            planeCoreY = Random.nextInt(boardSize - 2) + 1
            resultBoard[planeCoreX + 2][planeCoreY] = Cell.PLANE
            resultBoard[planeCoreX + 2][planeCoreY + 1] = Cell.PLANE
            resultBoard[planeCoreX + 2][planeCoreY - 1] = Cell.PLANE
        }

        Orientation.HeadingLeft -> {
            planeCoreX = Random.nextInt(boardSize - 2) + 1
            planeCoreY = Random.nextInt(boardSize - 3) + 1
            resultBoard[planeCoreX][planeCoreY + 2] = Cell.PLANE;
            resultBoard[planeCoreX - 1][planeCoreY + 2] = Cell.PLANE;
            resultBoard[planeCoreX + 1][planeCoreY + 2] = Cell.PLANE;
        }

        Orientation.HeadingDown -> {
            planeCoreX = Random.nextInt(boardSize - 3) + 2
            planeCoreY = Random.nextInt(boardSize - 2) + 1
            resultBoard[planeCoreX - 2][planeCoreY] = Cell.PLANE;
            resultBoard[planeCoreX - 2][planeCoreY + 1] = Cell.PLANE;
            resultBoard[planeCoreX - 2][planeCoreY - 1] = Cell.PLANE;
        }
    }
    // Populate the 'cross' in the plane
    resultBoard[planeCoreX][planeCoreY] = Cell.PLANE;
    resultBoard[planeCoreX + 1][planeCoreY] = Cell.PLANE;
    resultBoard[planeCoreX - 1][planeCoreY] = Cell.PLANE;
    resultBoard[planeCoreX][planeCoreY + 1] = Cell.PLANE;
    resultBoard[planeCoreX][planeCoreY - 1] = Cell.PLANE;

    return resultBoard
}
