cmake_minimum_required(VERSION 3.10)
project(DigitNN)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

set(SOURCES
    src/Matrix.cpp
    src/Layer.cpp
    src/Network.cpp
    src/DataLoader.cpp
)

# Training executable — with OpenMP, no Raylib
add_executable(Train
    main.cpp
    ${SOURCES}
)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Train OpenMP::OpenMP_CXX)
endif()

# Canvas executable — with Raylib, no OpenMP
add_executable(Canvas
    canvas_test.cpp
    ${SOURCES}
)
target_link_libraries(Canvas raylib)