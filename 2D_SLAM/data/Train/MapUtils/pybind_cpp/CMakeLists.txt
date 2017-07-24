cmake_minimum_required(VERSION 2.8.12)
project(map_util)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(PythonLibs 2 REQUIRED)
#MESSAGE(STATUS "${PYTHON_INCLUDE_DIRS}")
#MESSAGE(STATUS "${PYTHON_LIBRARIES}")

# export to bin
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../)

# Python binding
add_subdirectory(pybind11 "${CMAKE_CURRENT_BINARY_DIR}/pybind11_build")
pybind11_add_module(MapUtils.so src/map_utils_py.cpp)
pybind11_enable_warnings(MapUtils.so)
target_include_directories(MapUtils.so PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(MapUtils.so PRIVATE ${PYTHON_LIBRARIES})

#target_include_directories(MapUtils.so PRIVATE /cygdrive/c/Python27/include )
#target_link_libraries(MapUtils.so PRIVATE /cygdrive/c/Python27/libs/libpython27.a )



