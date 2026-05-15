# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22.1)
project(tokenizer_jni)

add_library(tokenizer_jni SHARED
    tokenizer_jni.cc
)

# In a public build, the headers for tokenizers-cpp should be provided.
target_include_directories(tokenizer_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Users should place libtokenizers_c.a built from mlc-ai/tokenizers-cpp here.
set(JNILIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/${ANDROID_ABI})

add_library(tokenizers_c STATIC IMPORTED)
set_target_properties(tokenizers_c PROPERTIES
    IMPORTED_LOCATION ${JNILIBS_DIR}/libtokenizers_c.a
)

target_link_libraries(tokenizer_jni
    tokenizers_c
    log
)