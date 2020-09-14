cmake_minimum_required(VERSION 3.17)
project(tutorial)

set(CMAKE_CXX_STANDARD 17)

add_executable(tutorial main.cpp Rectangle.h Rectangle.cpp Circle.cpp Circle.h Shape.h)

include_directories("/Users/sanchitnevgi/work/rl/lib")