cmake_minimum_required(VERSION 3.15)
project(postprocessor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Source files
set(POSTPROCESSOR_SRCS
    parser.c
    replacement.c
    schema.c
    core.c
)

# Static library
add_library(postprocessor_lib STATIC ${POSTPROCESSOR_SRCS})
target_include_directories(postprocessor_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Standalone CLI Executable
add_executable(postprocessor main.c)
target_link_libraries(postprocessor PRIVATE postprocessor_lib)

# Unit Test Executable
add_executable(test_postprocessor test_postprocessor.c)
target_link_libraries(test_postprocessor PRIVATE postprocessor_lib)

enable_testing()
add_test(NAME postprocessor_unit_tests COMMAND test_postprocessor)
