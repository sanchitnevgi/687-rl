cmake_minimum_required(VERSION 3.17)
project(HW7)

set(CMAKE_CXX_STANDARD 17)

include_directories("/Users/sanchitnevgi/work/reinforcement-learning/lib")

add_executable(hw7 main.cpp Agent.hpp Environment.hpp EpisodicAgent.cpp EpisodicAgent.hpp
        Gridworld.hpp Gridworld.cpp Manual.cpp Manual.hpp REINFORCE.cpp REINFORCE.hpp TabularBBO.cpp TabularBBO.hpp)