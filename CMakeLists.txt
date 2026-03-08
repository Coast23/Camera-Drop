cmake_minimum_required(VERSION 3.10)

project(Camera-Drop, CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES "src/codec/*.cpp" "src/util/*.cpp")

add_subdirectory(third_party/wirehair)
add_subdirectory(third_party/zstd/build/cmake zstd_build)

# correct 库在 Debug 模式下强行注入了 -fsanitize=address。
# 但 Windows 下的 MinGW 默认不包含 libasan，会导致链接失败。
# 为了规避这个问题，暂时先将 MinGW 下的 BUILD 模式改为 Release
if(MINGW)
    set(BACKUP_BUILD_TYPE ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE "Release")
    add_subdirectory(third_party/correct)
    set(CMAKE_BUILD_TYPE ${BACKUP_BUILD_TYPE})
else()
    add_subdirectory(third_party/correct)
endif()

# OpenCV 请自备
find_package(OpenCV REQUIRED)

add_executable(debug src/test/test_codec.cpp ${SRC_FILES})
# add_executable(debug src/test/test_img_writer.cpp ${SRC_FILES})

target_include_directories(debug PRIVATE
    src
    third_party/correct/include # correct.h 需指定路径
)

target_link_libraries(debug PRIVATE
    wirehair
    correct_static
    libzstd_static
    ${OpenCV_LIBS}
)