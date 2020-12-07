cmake_minimum_required(VERSION 3.17)
project(Homework6)

set(CMAKE_CXX_STANDARD 17)

include_directories("/Users/sanchitnevgi/work/reinforcement-learning/lib")

add_executable(Homework6
        main.cpp
        Environment.hpp
        Agent.hpp
        Gridworld.cpp
        Gridworld.hpp
        CartPole.cpp
        CartPole.hpp
        MountainCar.cpp
        MountainCar.hpp
        CartPole.hpp
        CartPole.cpp
        QLearning.cpp
        QLearning.hpp
        Sarsa.cpp
        Sarsa.hpp)