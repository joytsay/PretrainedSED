cmake_minimum_required(VERSION 3.22)
project(atst_sed_testbed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)

option(BUILD_QT_TESTBED "Build the original Qt playlist testbed" ON)
if(BUILD_QT_TESTBED)
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
endif()

add_executable(atst_sed_web_testbed
    src/web_testbed_server.cpp
)
target_link_libraries(atst_sed_web_testbed PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
)

option(BUILD_SVELTE_UI "Build the Svelte web UI with the web testbed" ON)
if(BUILD_SVELTE_UI)
    find_program(NPM_EXECUTABLE npm)
    if(NPM_EXECUTABLE)
        file(GLOB_RECURSE SED_WEB_UI_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/src/webui/src/*
            ${CMAKE_CURRENT_SOURCE_DIR}/src/webui/package.json
            ${CMAKE_CURRENT_SOURCE_DIR}/src/webui/package-lock.json
            ${CMAKE_CURRENT_SOURCE_DIR}/src/webui/svelte.config.js
            ${CMAKE_CURRENT_SOURCE_DIR}/src/webui/vite.config.ts
            ${CMAKE_CURRENT_SOURCE_DIR}/src/webui/tsconfig.json
        )
        set(SED_WEB_UI_OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/webui/dist/index.html)
        add_custom_command(
            OUTPUT ${SED_WEB_UI_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E env NPM_CONFIG_AUDIT=false NPM_CONFIG_FUND=false
                    sh -c "npm ci && npm run build"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/webui
            DEPENDS ${SED_WEB_UI_SOURCES}
            COMMENT "Building GeoVision SED Svelte UI"
            VERBATIM
        )
        add_custom_target(atst_sed_webui DEPENDS ${SED_WEB_UI_OUTPUT})
        add_dependencies(atst_sed_web_testbed atst_sed_webui)
    else()
        message(WARNING "npm not found; atst_sed_web_testbed will use the existing src/webui/dist build")
    endif()
endif()

option(BUILD_TENSORRT_WORKER "Build the TensorRT callback worker" ON)
if(BUILD_TENSORRT_WORKER)
    find_package(CUDAToolkit REQUIRED)
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

install(TARGETS atst_sed_web_testbed RUNTIME DESTINATION bin)
if(TARGET atst_sed_testbed)
    install(TARGETS atst_sed_testbed RUNTIME DESTINATION bin)
endif()
if(TARGET atst_sed_worker)
    install(TARGETS atst_sed_worker RUNTIME DESTINATION bin)
endif()
