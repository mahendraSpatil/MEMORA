cmake_minimum_required(VERSION 3.16)
project(memora CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(memora_core
  src/memory.cpp
  src/graph.cpp
  src/tarjan.cpp
  src/scoring.cpp
  src/generator.cpp
  src/policies.cpp
  src/metrics.cpp
  src/knapsack.cpp
  src/benchmark.cpp
)
target_include_directories(memora_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(memora_core PRIVATE /W3)
else()
  target_compile_options(memora_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(memora src/main.cpp)
target_link_libraries(memora PRIVATE memora_core)

add_executable(memora_tests tests/test_main.cpp)
target_link_libraries(memora_tests PRIVATE memora_core)

enable_testing()
add_test(NAME validation COMMAND memora_tests)
