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
set(CAMDROP_FFMPEG_EXE "" CACHE FILEPATH "Optional ffmpeg executable to copy next to built tools")
set(CAMDROP_RESOURCE_DIR "" CACHE PATH "Root directory containing web/model/ and cnn/models/. Defaults to CMAKE_SOURCE_DIR.")

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
    src/vision/pattern_cnn.cpp
    src/vision/yolo_localizer.cpp
)

set(CAMDROP_RECT_LAYOUT_DEFS
    CAMDROP_CFG_MARGIN=9
    CAMDROP_CFG_SHORT_EDGE_PATTERNS=130
)

add_subdirectory(third_party/wirehair)
add_subdirectory(third_party/zstd/build/cmake zstd_build)

if(MINGW)
    set(_camdrop_saved_build_type "${CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE "Release")
    add_subdirectory(third_party/correct)
    set(CMAKE_BUILD_TYPE "${_camdrop_saved_build_type}")
else()
    add_subdirectory(third_party/correct)
endif()

set(_opencv_via_pkgconfig FALSE)

if(MINGW)
    find_package(OpenCV QUIET)
    if(NOT OpenCV_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(OPENCV4 REQUIRED IMPORTED_TARGET opencv4)
        set(OpenCV_FOUND TRUE)
        set(OpenCV_INCLUDE_DIRS ${OPENCV4_INCLUDE_DIRS})
        set(OpenCV_LIBRARY_DIRS ${OPENCV4_STATIC_LIBRARY_DIRS})
        set(OpenCV_LIBS ${OPENCV4_STATIC_LIBRARIES})
        set(_opencv_via_pkgconfig TRUE)
    endif()
else()
    find_package(OpenCV REQUIRED)
endif()

set(CAMDROP_OPENCV_EXTRA_LIBS "")
if(_opencv_via_pkgconfig)
    list(APPEND CAMDROP_OPENCV_EXTRA_LIBS
        jpeg png tiff webp webpdecoder webpdemux webpmux sharpyuv zlib lzma
    )
endif()

add_library(camdrop_opencv INTERFACE)
target_include_directories(camdrop_opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
if(OpenCV_LIBRARY_DIRS)
    target_link_directories(camdrop_opencv INTERFACE ${OpenCV_LIBRARY_DIRS})
endif()
if(_opencv_via_pkgconfig)
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
    get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    unset(ONNXRUNTIME_DLL CACHE)

    # Prefer a DLL adjacent to the selected import library to avoid version mismatch.
    find_file(ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT ${ONNXRUNTIME_LIBRARY_DIR}
        PATH_SUFFIXES bin lib
        NO_DEFAULT_PATH
    )
    if(NOT ONNXRUNTIME_DLL)
        find_file(ONNXRUNTIME_DLL
            NAMES onnxruntime.dll
            HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT ${ONNXRUNTIME_LIBRARY_DIR}
            PATH_SUFFIXES bin lib
        )
    endif()
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
        target_compile_options(${target_name} PRIVATE /W3 /permissive-)
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

function(camdrop_copy_opencv_dll target_name)
    if(WIN32 AND NOT MINGW)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:${target_name}>
                    $<TARGET_FILE_DIR:${target_name}>
                COMMAND_EXPAND_LISTS
            )
        endif()
    endif()
endfunction()

function(camdrop_copy_ffmpeg_binary target_name)
    if(WIN32 AND CAMDROP_FFMPEG_EXE)
        if(EXISTS "${CAMDROP_FFMPEG_EXE}")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CAMDROP_FFMPEG_EXE}"
                    "$<TARGET_FILE_DIR:${target_name}>/ffmpeg.exe"
            )
        else()
            message(WARNING "CAMDROP_FFMPEG_EXE does not exist: ${CAMDROP_FFMPEG_EXE}")
        endif()
    endif()
endfunction()

# camdrop_copy_resource_dir(target): copies model subdirectories from CAMDROP_RESOURCE_DIR
# (web/model and cnn/models) into the binary output dir, preserving relative paths.
# Usage: camdrop_copy_resource_dir(my_target)
function(camdrop_copy_resource_dir target_name)
    if(NOT CAMDROP_RESOURCE_DIR)
        return()
    endif()
    if(NOT EXISTS "${CAMDROP_RESOURCE_DIR}")
        message(WARNING "CAMDROP_RESOURCE_DIR does not exist: ${CAMDROP_RESOURCE_DIR}")
        return()
    endif()
    foreach(_subdir web/model cnn/models pattern_finder/best_v2)
        set(_src "${CAMDROP_RESOURCE_DIR}/${_subdir}")
        if(EXISTS "${_src}")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                    "${_src}"
                    "$<TARGET_FILE_DIR:${target_name}>/${_subdir}"
                COMMENT "Copying ${_subdir} to output dir"
            )
        endif()
    endforeach()
endfunction()

add_library(camdrop_core INTERFACE)
target_include_directories(camdrop_core
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/correct/include
)
target_link_libraries(camdrop_core
    INTERFACE
        wirehair correct_static libzstd_static camdrop_opencv
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
    camdrop_copy_ffmpeg_binary(file_video_encoder)

    if(CAMDROP_BUILD_ORT_PIPELINE)
        add_library(camdrop_vision_ort STATIC ${VISION_ORT_SRC})
        camdrop_configure_target(camdrop_vision_ort)
        target_link_libraries(camdrop_vision_ort
            PUBLIC
                camdrop_vision_common camdrop_opencv onnxruntime::onnxruntime
        )

        add_executable(frame_scanner src/misc/frame_scanner.cpp)
        camdrop_configure_target(frame_scanner)
        target_link_libraries(frame_scanner PRIVATE camdrop_vision_ort camdrop_opencv)
        camdrop_copy_onnxruntime_dll(frame_scanner)
        camdrop_copy_resource_dir(frame_scanner)

        add_executable(anchor_scanner src/misc/anchor_scanner.cpp)
        camdrop_configure_target(anchor_scanner)
        target_link_libraries(anchor_scanner PRIVATE onnxruntime::onnxruntime camdrop_opencv)
        camdrop_copy_onnxruntime_dll(anchor_scanner)
        camdrop_copy_resource_dir(anchor_scanner)

        add_executable(file_video_decoder src/misc/file_video_decoder.cpp)
        camdrop_configure_target(file_video_decoder)
        target_link_libraries(file_video_decoder PRIVATE camdrop_core camdrop_vision_ort camdrop_opencv)
        add_dependencies(file_video_decoder wirehair correct_static libzstd_static)
        camdrop_copy_onnxruntime_dll(file_video_decoder)
        camdrop_copy_ffmpeg_binary(file_video_decoder)
        camdrop_copy_resource_dir(file_video_decoder)

        add_executable(realshot_compare src/misc/realshot_compare.cpp)
        camdrop_configure_target(realshot_compare)
        target_link_libraries(realshot_compare PRIVATE camdrop_vision_ort camdrop_opencv)
        camdrop_copy_onnxruntime_dll(realshot_compare)
        camdrop_copy_resource_dir(realshot_compare)

        add_executable(recognizer_diff src/misc/recognizer_diff.cpp)
        camdrop_configure_target(recognizer_diff)
        target_link_libraries(recognizer_diff PRIVATE camdrop_vision_ort camdrop_opencv)
        camdrop_copy_onnxruntime_dll(recognizer_diff)
        camdrop_copy_resource_dir(recognizer_diff)

        add_library(camdrop_vision_common_rect STATIC ${VISION_COMMON_SRC})
        camdrop_configure_target(camdrop_vision_common_rect)
        target_compile_definitions(camdrop_vision_common_rect PRIVATE ${CAMDROP_RECT_LAYOUT_DEFS})
        target_link_libraries(camdrop_vision_common_rect PUBLIC camdrop_opencv)

        add_library(camdrop_vision_ort_rect STATIC ${VISION_ORT_SRC})
        camdrop_configure_target(camdrop_vision_ort_rect)
        target_compile_definitions(camdrop_vision_ort_rect PRIVATE ${CAMDROP_RECT_LAYOUT_DEFS})
        target_link_libraries(camdrop_vision_ort_rect
            PUBLIC
                camdrop_vision_common_rect camdrop_opencv onnxruntime::onnxruntime
        )

        add_executable(rect_layout_probe src/misc/rect_layout_probe.cpp)
        camdrop_configure_target(rect_layout_probe)
        target_compile_definitions(rect_layout_probe PRIVATE ${CAMDROP_RECT_LAYOUT_DEFS})
        target_link_libraries(rect_layout_probe PRIVATE camdrop_vision_ort_rect camdrop_opencv)
        camdrop_copy_onnxruntime_dll(rect_layout_probe)
        camdrop_copy_resource_dir(rect_layout_probe)
    endif()
endif()

if(CAMDROP_BUILD_MISC_TOOLS)
    add_executable(pattern_generator src/misc/pattern_generator.cpp)
    camdrop_configure_target(pattern_generator)
    target_link_libraries(pattern_generator PRIVATE camdrop_core)

    add_executable(fullscreen_player src/misc/fullscreen_player.cpp)
    camdrop_configure_target(fullscreen_player)
    target_link_libraries(fullscreen_player PRIVATE camdrop_opencv)
    if(WIN32)
        target_link_libraries(fullscreen_player PRIVATE d3d11 dxgi d3dcompiler dwmapi winmm avrt ole32)
    endif()

    add_executable(config_acc_test src/misc/config_acc_test.cpp)
    camdrop_configure_target(config_acc_test)
    target_link_libraries(config_acc_test PRIVATE camdrop_opencv)

    add_executable(img_parser src/misc/img_parser.cpp)
    camdrop_configure_target(img_parser)
    target_link_libraries(img_parser PRIVATE camdrop_core camdrop_opencv)
endif()
