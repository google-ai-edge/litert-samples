# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sample-local JNI workaround for CompiledModel dynamic shapes; resolves the
# LiteRT C API symbols from the litert AAR's libLiteRt.so at runtime, so no
# LiteRT headers or import libraries are needed at build time.
cmake_minimum_required(VERSION 3.22.1)
project(dynamic_shape_jni CXX)

add_library(dynamic_shape_jni SHARED dynamic_shape_jni.cc)
target_compile_options(dynamic_shape_jni PRIVATE -Wall -Werror)
# 16 KB page-size devices require 16 KB-aligned ELF LOAD segments.
target_link_options(dynamic_shape_jni PRIVATE "-Wl,-z,max-page-size=16384")
