cmake_minimum_required(VERSION 3.17)
project(hw2)

set(CMAKE_CXX_STANDARD 17)

include_directories(code)

add_executable(hw2
        code/Agent.hpp
        code/Environment.hpp
        code/EpisodicAgent.cpp
        code/EpisodicAgent.hpp
        code/Gridworld.cpp
        code/Gridworld.hpp
        code/main.cpp
        code/Manual.cpp
        code/Manual.hpp
        code/TabularBBO.cpp
        code/TabularBBO.hpp
        code/TabularRandomSearch.cpp
        code/TabularRandomSearch.hpp)

include_directories("/Users/sanchitnevgi/work/reinforcement-learning/lib")
