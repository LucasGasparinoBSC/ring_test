cmake_minimum_required(VERSION 3.21)

project(ring_test LANGUAGES CXX)

# Set compiler to mpicxx
set(ENV{CXX} mpicxx)

find_package(MPI)
if (NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "MPI CXX compiler not found")
endif()

# Set CXX standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If not set, default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set library type to shared
set(BUILD_SHARED_LIBS ON)
set(LIBRARY_TYPE SHARED)

# CXX flags (GCC)
set(CMAKE_CXX_FLAGS "-mcmodel=small -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Add subdirectory
add_subdirectory(src)