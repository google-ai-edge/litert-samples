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

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val LightPalette =
  lightColorScheme(
    primary = Color(0xFF006A63),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFC3EEE6),
    onPrimaryContainer = Color(0xFF004B46),
    secondary = Color(0xFF526662),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFD5E9E3),
    onSecondaryContainer = Color(0xFF263B36),
    tertiary = Color(0xFF53647E),
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFD7E3FD),
    onTertiaryContainer = Color(0xFF304761),
    background = Color(0xFFF6F8F7),
    onBackground = Color(0xFF182522),
    surface = Color(0xFFFCFEFC),
    onSurface = Color(0xFF182522),
    surfaceVariant = Color(0xFFE0E9E4),
    onSurfaceVariant = Color(0xFF465550),
    surfaceTint = Color(0xFF006A63),
    surfaceBright = Color(0xFFFCFEFC),
    surfaceDim = Color(0xFFD8E1DC),
    surfaceContainerLowest = Color.White,
    surfaceContainerLow = Color(0xFFF0F5F1),
    surfaceContainer = Color(0xFFECF2ED),
    surfaceContainerHigh = Color(0xFFE7EEEA),
    surfaceContainerHighest = Color(0xFFE1E9E4),
    inverseSurface = Color(0xFF2B352F),
    inverseOnSurface = Color(0xFFF0F5F1),
    inversePrimary = Color(0xFF80D5C7),
    scrim = Color.Black,
    outline = Color(0xFF73837B),
    outlineVariant = Color(0xFFCFDAD3),
    error = Color(0xFFBA1A1A),
    onError = Color.White,
    errorContainer = Color(0xFFFFDAD6),
    onErrorContainer = Color(0xFF93000A),
  )

private val DarkPalette =
  darkColorScheme(
    primary = Color(0xFF80D5C7),
    onPrimary = Color(0xFF003731),
    primaryContainer = Color(0xFF005048),
    onPrimaryContainer = Color(0xFF9CF2E3),
    secondary = Color(0xFFB5CCC4),
    onSecondary = Color(0xFF203630),
    secondaryContainer = Color(0xFF374C45),
    onSecondaryContainer = Color(0xFFD1E8DF),
    tertiary = Color(0xFFBAC8E4),
    onTertiary = Color(0xFF24314B),
    tertiaryContainer = Color(0xFF3B4863),
    onTertiaryContainer = Color(0xFFD7E3FD),
    background = Color(0xFF101916),
    onBackground = Color(0xFFDFE8E1),
    surface = Color(0xFF15201C),
    onSurface = Color(0xFFDFE8E1),
    surfaceVariant = Color(0xFF3E4B44),
    onSurfaceVariant = Color(0xFFBDC9C1),
    surfaceTint = Color(0xFF80D5C7),
    surfaceBright = Color(0xFF35443C),
    surfaceDim = Color(0xFF101916),
    surfaceContainerLowest = Color(0xFF0B120F),
    surfaceContainerLow = Color(0xFF15201C),
    surfaceContainer = Color(0xFF19251F),
    surfaceContainerHigh = Color(0xFF233029),
    surfaceContainerHighest = Color(0xFF2E3B34),
    inverseSurface = Color(0xFFDFE8E1),
    inverseOnSurface = Color(0xFF19251F),
    inversePrimary = Color(0xFF006A63),
    scrim = Color.Black,
    outline = Color(0xFF88958D),
    outlineVariant = Color(0xFF3E4B44),
    error = Color(0xFFFFB4AB),
    onError = Color(0xFF690005),
    errorContainer = Color(0xFF93000A),
    onErrorContainer = Color(0xFFFFDAD6),
  )

private fun type(size: Int, lineHeight: Int, weight: FontWeight = FontWeight.Normal) =
  TextStyle(
    fontFamily = FontFamily.SansSerif,
    fontWeight = weight,
    fontSize = size.sp,
    lineHeight = lineHeight.sp,
  )

private val AppTypography =
  Typography(
    displayLarge = type(48, 56),
    displayMedium = type(40, 48),
    displaySmall = type(32, 40),
    headlineLarge = type(30, 38),
    headlineMedium = type(26, 34),
    headlineSmall = type(24, 32),
    titleLarge = type(22, 28, FontWeight.SemiBold),
    titleMedium = type(16, 22, FontWeight.SemiBold),
    titleSmall = type(14, 20, FontWeight.SemiBold),
    bodyLarge = type(16, 24),
    bodyMedium = type(14, 20),
    bodySmall = type(12, 16),
    labelLarge = type(14, 20, FontWeight.Medium),
    labelMedium = type(12, 16, FontWeight.Medium),
    labelSmall = type(11, 16, FontWeight.Medium),
  )

/** One static palette per system theme; model results never depend on wallpaper colors. */
@Composable
fun ApplicationTheme(content: @Composable () -> Unit) {
  MaterialTheme(
    colorScheme = if (isSystemInDarkTheme()) DarkPalette else LightPalette,
    typography = AppTypography,
    shapes =
      Shapes(
        extraSmall = RoundedCornerShape(4.dp),
        small = RoundedCornerShape(8.dp),
        medium = RoundedCornerShape(12.dp),
        large = RoundedCornerShape(16.dp),
        extraLarge = RoundedCornerShape(24.dp),
      ),
    content = content,
  )
}
