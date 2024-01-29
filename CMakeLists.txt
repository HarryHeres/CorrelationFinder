cmake_minimum_required(VERSION 3.13) 

project(PPR)            

set(OUTPUT_BINARY ppr)

file(GLOB_RECURSE sources src/*.cpp src/headers/*.hpp)
file(GLOB_RECURSE sources_test src/test/*.cpp)
file(GLOB_RECURSE data resources/*)

set(CMAKE_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_OUTPUT_DIR}/exec)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(APPLE)
  add_link_options("-Wl,-ld_classic") # This is here due to https://github.com/Homebrew/homebrew-core/issues/145991
endif()

add_link_options("-ltbb") 

find_package(OpenCL)

add_executable(${OUTPUT_BINARY} ${sources})
target_link_libraries(${OUTPUT_BINARY} OpenCL::OpenCL)

# target_compile_options(${OUTPUT_BINARY} PRIVATE -g -std=c++17 -Wall -Wextra -Wfloat-conversion -pedantic) # DEBUG
target_compile_options(${OUTPUT_BINARY} PRIVATE -std=c++17 -Wall -Wextra -Wfloat-conversion -pedantic -O3) # RELEASE
