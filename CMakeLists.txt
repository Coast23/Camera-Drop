cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(CameraDrop LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CAMDROP_BUILD_VISION "Build vision modules" ON)
option(CAMDROP_BUILD_ORT_PIPELINE "Build ONNX Runtime YOLO pipeline tools" ON)
option(CAMDROP_BUILD_MISC_TOOLS "Build misc command line tools" ON)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of an ONNX Runtime installation")

file(GLOB_RECURSE CORE_SRC_FILES CONFIGURE_DEPENDS
    "src/codec/*.cpp"
    "src/util/*.cpp"
)

set(VISION_COMMON_SRC
    src/vision/deskew.cpp
    src/vision/frame_renderer.cpp
    src/vision/pattern_dict.cpp
    src/vision/recognizer.cpp
    src/vision/visual_frame_codec.cpp
)

set(VISION_ORT_SRC
    src/vision/frame_pipeline.cpp
    src/vision/yolo_localizer.cpp
)

add_subdirectory(third_party/wirehair)
add_subdirectory(third_party/zstd/build/cmake zstd_build)

# correct 库在 Debug 模式下强行注入了 -fsanitize=address。
# Windows 下的 MinGW 通常没有对应的 libasan，单独降级 correct 的构建类型。
if(MINGW)
    set(_camdrop_saved_build_type "${CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE "Release")
    add_subdirectory(third_party/correct)
    set(CMAKE_BUILD_TYPE "${_camdrop_saved_build_type}")
else()
    add_subdirectory(third_party/correct)
endif()

if(MINGW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENCV4 REQUIRED IMPORTED_TARGET opencv4)
    set(OpenCV_FOUND TRUE)
    set(OpenCV_INCLUDE_DIRS ${OPENCV4_INCLUDE_DIRS})
    set(OpenCV_LIBRARY_DIRS ${OPENCV4_STATIC_LIBRARY_DIRS})
    set(OpenCV_LIBS ${OPENCV4_STATIC_LIBRARIES})
else()
    find_package(OpenCV REQUIRED)
endif()

set(CAMDROP_OPENCV_EXTRA_LIBS "")
if(MINGW)
    list(APPEND CAMDROP_OPENCV_EXTRA_LIBS
        jpeg
        png
        tiff
        webp
        webpdecoder
        webpdemux
        webpmux
        sharpyuv
        zlib
        lzma
    )
endif()

add_library(camdrop_opencv INTERFACE)
target_include_directories(camdrop_opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
if(OpenCV_LIBRARY_DIRS)
    target_link_directories(camdrop_opencv INTERFACE ${OpenCV_LIBRARY_DIRS})
endif()
if(MINGW)
    target_link_libraries(camdrop_opencv
        INTERFACE
            "$<LINK_GROUP:RESCAN,${OpenCV_LIBS},${CAMDROP_OPENCV_EXTRA_LIBS}>"
    )
else()
    target_link_libraries(camdrop_opencv INTERFACE ${OpenCV_LIBS})
endif()

if(CAMDROP_BUILD_ORT_PIPELINE)
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
        PATH_SUFFIXES include
    )
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime onnxruntime.lib
        HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
        PATH_SUFFIXES lib lib64
    )
    find_file(ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
        PATH_SUFFIXES bin lib
    )
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(ONNXRuntime
        REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY
    )
    if(NOT ONNXRuntime_FOUND)
        message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT to its installation directory.")
    endif()
    if(NOT TARGET onnxruntime::onnxruntime)
        add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION             "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
        if(WIN32 AND ONNXRUNTIME_DLL)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_IMPLIB  "${ONNXRUNTIME_LIBRARY}"
                IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
            )
        endif()
    endif()
    mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY ONNXRUNTIME_DLL)
endif()

function(camdrop_configure_target target_name)
    target_include_directories(${target_name}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/correct/include
            ${OpenCV_INCLUDE_DIRS}
    )

    if(OpenCV_LIBRARY_DIRS)
        target_link_directories(${target_name} PUBLIC ${OpenCV_LIBRARY_DIRS})
    endif()

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W3)
    else()
        target_compile_options(${target_name} PRIVATE -O2 -Wall)
    endif()
endfunction()

function(camdrop_copy_onnxruntime_dll target_name)
    if(WIN32 AND CAMDROP_BUILD_ORT_PIPELINE AND ONNXRUNTIME_DLL)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}"
                "$<TARGET_FILE_DIR:${target_name}>"
        )
    endif()
endfunction()

add_library(camdrop_core INTERFACE)
target_include_directories(camdrop_core
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/correct/include
)
target_link_libraries(camdrop_core
    INTERFACE
        wirehair
        correct_static
        libzstd_static
        camdrop_opencv
)

add_executable(debug src/test/test_codec.cpp)
camdrop_configure_target(debug)
target_link_libraries(debug PRIVATE camdrop_core)

if(CAMDROP_BUILD_VISION)
    add_library(camdrop_vision_common STATIC ${VISION_COMMON_SRC})
    camdrop_configure_target(camdrop_vision_common)
    target_link_libraries(camdrop_vision_common PUBLIC camdrop_opencv)

    add_executable(recognizer_debug src/misc/recognizer_debug.cpp)
    camdrop_configure_target(recognizer_debug)
    target_link_libraries(recognizer_debug PRIVATE camdrop_vision_common camdrop_opencv)

    add_executable(file_video_encoder src/misc/file_video_encoder.cpp)
    camdrop_configure_target(file_video_encoder)
    target_link_libraries(file_video_encoder PRIVATE camdrop_core camdrop_vision_common camdrop_opencv)
    add_dependencies(file_video_encoder wirehair correct_static libzstd_static)

    if(CAMDROP_BUILD_ORT_PIPELINE)
        add_library(camdrop_vision_ort STATIC ${VISION_ORT_SRC})
        camdrop_configure_target(camdrop_vision_ort)
        target_link_libraries(camdrop_vision_ort
            PUBLIC
                camdrop_vision_common
                camdrop_opencv
                onnxruntime::onnxruntime
        )

        add_executable(frame_scanner src/misc/frame_scanner.cpp)
        camdrop_configure_target(frame_scanner)
        target_link_libraries(frame_scanner PRIVATE camdrop_vision_ort camdrop_opencv)
        camdrop_copy_onnxruntime_dll(frame_scanner)

        add_executable(anchor_scanner src/misc/anchor_scanner.cpp)
        camdrop_configure_target(anchor_scanner)
        target_link_libraries(anchor_scanner PRIVATE onnxruntime::onnxruntime camdrop_opencv)
        camdrop_copy_onnxruntime_dll(anchor_scanner)

        add_executable(file_video_decoder src/misc/file_video_decoder.cpp)
        camdrop_configure_target(file_video_decoder)
        target_link_libraries(file_video_decoder PRIVATE camdrop_core camdrop_vision_ort camdrop_opencv)
        add_dependencies(file_video_decoder wirehair correct_static libzstd_static)
        camdrop_copy_onnxruntime_dll(file_video_decoder)

        add_executable(realshot_compare src/misc/realshot_compare.cpp)
        camdrop_configure_target(realshot_compare)
        target_link_libraries(realshot_compare PRIVATE camdrop_vision_ort camdrop_opencv)
        camdrop_copy_onnxruntime_dll(realshot_compare)

        add_executable(recognizer_diff src/misc/recognizer_diff.cpp)
        camdrop_configure_target(recognizer_diff)
        target_link_libraries(recognizer_diff PRIVATE camdrop_vision_ort camdrop_opencv)
        camdrop_copy_onnxruntime_dll(recognizer_diff)
    endif()
endif()

if(CAMDROP_BUILD_MISC_TOOLS)
    add_executable(pattern_generator src/misc/pattern_generator.cpp)
    camdrop_configure_target(pattern_generator)
    target_link_libraries(pattern_generator PRIVATE camdrop_core)

    add_executable(config_acc_test src/misc/config_acc_test.cpp)
    camdrop_configure_target(config_acc_test)
    target_link_libraries(config_acc_test PRIVATE camdrop_opencv)

    add_executable(img_parser src/misc/img_parser.cpp)
    camdrop_configure_target(img_parser)
    target_link_libraries(img_parser PRIVATE camdrop_core camdrop_opencv)
endif()
