cmake_minimum_required(VERSION 3.20)
project(silicon_autopsy LANGUAGES CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "SILICON AUTOPSY targets Apple Silicon (Metal) only.")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)

# --- Metal kernels -> metallib (offline compiled, so the exact compiler
#     version is fixed at build time and recorded with every session).
set(KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernels/memory.metal)
set(METALLIB ${CMAKE_BINARY_DIR}/silicon_autopsy.metallib)
add_custom_command(
  OUTPUT ${METALLIB}
  COMMAND xcrun -sdk macosx metal -O2 -Wall -Werror
          -I ${CMAKE_SOURCE_DIR}/kernels -o ${METALLIB} ${KERNEL_SRC}
  DEPENDS ${KERNEL_SRC} ${CMAKE_SOURCE_DIR}/kernels/shared_types.h
  COMMENT "Compiling Metal kernels"
  VERBATIM)
add_custom_target(kernels ALL DEPENDS ${METALLIB})

execute_process(COMMAND xcrun -sdk macosx metal --version
                OUTPUT_VARIABLE METAL_VERSION_RAW ERROR_QUIET)
string(REGEX MATCH "^[^\n]*" METAL_VERSION "${METAL_VERSION_RAW}")

# --- Core library (pure C++, unit-testable without a GPU)
add_library(sa_core STATIC src/config.cpp src/workloads.cpp)
target_include_directories(sa_core PUBLIC src)
target_compile_options(sa_core PRIVATE -Wall -Wextra)

# --- Benchmark executable
add_executable(silicon-autopsy src/main.mm src/metal_ctx.mm src/record.mm)
add_dependencies(silicon-autopsy kernels)
target_link_libraries(silicon-autopsy PRIVATE sa_core
  "-framework Metal" "-framework Foundation" "-framework IOKit" "-framework CoreFoundation")
target_compile_options(silicon-autopsy PRIVATE -Wall -Wextra -fobjc-arc)
target_compile_definitions(silicon-autopsy PRIVATE
  SA_METALLIB_PATH="${METALLIB}"
  SA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  SA_METAL_VERSION="${METAL_VERSION}")

# --- Unit tests (ctest --test-dir build)
enable_testing()
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE sa_core)
target_compile_options(test_engine PRIVATE -Wall -Wextra)
add_test(NAME engine_unit_tests COMMAND test_engine)
