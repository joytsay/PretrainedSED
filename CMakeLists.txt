cmake_minimum_required(VERSION 3.22)
project(atst_sed_testbed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 6.4 REQUIRED COMPONENTS Widgets Multimedia MultimediaWidgets)

add_executable(atst_sed_testbed
    src/testbed_main.cpp
)
target_link_libraries(atst_sed_testbed PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::MultimediaWidgets
)

option(BUILD_TENSORRT_WORKER "Build the TensorRT callback worker" ON)
if(BUILD_TENSORRT_WORKER)
    find_package(CUDAToolkit REQUIRED)
    find_package(Threads REQUIRED)
    find_package(nlohmann_json 3.10 REQUIRED)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/local/TensorRT/include)
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/TensorRT/lib)
    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
        message(FATAL_ERROR "TensorRT headers/library not found; install TensorRT development packages or set TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARY")
    endif()

    add_executable(atst_sed_worker
        src/sed_worker.cpp
    )
    target_include_directories(atst_sed_worker PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(atst_sed_worker PRIVATE
        ${TENSORRT_LIBRARY}
        CUDA::cudart
        Threads::Threads
        nlohmann_json::nlohmann_json
    )
endif()

install(TARGETS atst_sed_testbed
    RUNTIME DESTINATION bin)
if(TARGET atst_sed_worker)
    install(TARGETS atst_sed_worker RUNTIME DESTINATION bin)
endif()
