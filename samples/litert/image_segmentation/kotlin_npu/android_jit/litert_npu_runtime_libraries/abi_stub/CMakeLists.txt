cmake_minimum_required(VERSION 3.22.1)

project(litert_abi_stub C)

if(NOT DEFINED LITERT_ABI_STUB_NAME)
  message(FATAL_ERROR "LITERT_ABI_STUB_NAME must be set")
endif()

add_library(${LITERT_ABI_STUB_NAME} SHARED abi_stub.c)