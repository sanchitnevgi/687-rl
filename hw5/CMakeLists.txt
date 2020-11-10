cmake_minimum_required(VERSION 3.17)
project(hw5)

set(CMAKE_CXX_STANDARD 17)

include_directories(code)

add_executable(hw5
        code/main.cpp
        code/Agent.hpp
        code/CartPole.cpp
        code/CartPole.hpp
        code/Environment.hpp
        code/Gridworld.cpp
        code/Gridworld.hpp
        code/MountainCar.cpp
        code/MountainCar.hpp
        code/QLearning.cpp
        code/QLearning.hpp
        code/Sarsa.cpp
        code/Sarsa.hpp)

include_directories("/Users/sanchitnevgi/work/reinforcement-learning/lib")