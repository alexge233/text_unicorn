cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-std=c++1y")
SET(EIGEN3_INCLUDE_DIR "/usr/include/eigen3/")
IF(NOT EIGEN3_INCLUDE_DIR)
    MESSAGE( FATAL_ERROR "Set EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 location")
ENDIF()
INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")
# This is used for building unit tests only!!!
# We use the brand new and shiny Catch C++ 2.0: 
# https://github.com/catchorg/Catch2/releases/download/v2.0.1/catch.hpp
if(EXISTS "${PROJECT_SOURCE_DIR}/catch.hpp")
	message(STATUS "header `catch.hpp` already exists, not downloading.")
else()
	set(wget_command "wget")
	set(wget_arg "https://github.com/catchorg/Catch2/releases/download/v2.0.1/catch.hpp")
	message(STATUS "fetching catch.hpp single header.")
	execute_process(COMMAND ${wget_command} ${wget_arg}
					WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/)
endif()
set(CMAKE_BUILD_TYPE Debug)
set(TESTFILE ${TESTFILE} unit_test.cpp)
add_executable(test_the_rainbow ${TESTFILE})
set_property(TARGET test_the_rainbow PROPERTY CXX_STANDARD 14)
