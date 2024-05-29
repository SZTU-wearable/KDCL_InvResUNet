cmake_minimum_required(VERSION 3.10)
project(GetECGPPGData)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# 如果mpdev库和头文件不在标准位置，请使用include_directories和link_directories指定它们的位置
include_directories(${CMAKE_SOURCE_DIR})
set(LIBRARY_PATH "${CMAKE_SOURCE_DIR}/x64")
link_directories(${LIBRARY_PATH})

add_library(get_ecg_ppg_data SHARED get_ecg_ppg_data.cpp)

## 链接mpdev库
target_link_libraries(get_ecg_ppg_data ${LIBRARY_PATH}/mpdev.lib)
#
add_custom_command(TARGET get_ecg_ppg_data POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LIBRARY_PATH}/mpdev.dll"
        "${LIBRARY_PATH}/xerces-c_3_1.dll"
        $<TARGET_FILE_DIR:get_ecg_ppg_data>)
