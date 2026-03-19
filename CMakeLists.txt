cmake_minimum_required(VERSION 3.10)
project(DigitNN)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(DigitNN
    main.cpp
    src/Matrix.cpp
    src/Layer.cpp
    src/Network.cpp
    src/DataLoader.cpp
)

target_link_libraries(DigitNN raylib)