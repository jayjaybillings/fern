#------------------------------------------------------------------------------
# Copyright (c) 2015-, The University of Tennessee
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of fern nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Author(s): Jay Jay Billings, Ben Brock, Andrew Belt, Dan Shyles
#----------------------------------------------------------------------------*/

# Author: Jay Jay Billings
# Date: 20150907

# Set a name for this project.
PROJECT ( FERN CXX )
# Set the minimum required version of cmake for this project.
cmake_minimum_required(VERSION 2.8)
# Configure some default options
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)
set(CMAKE_CXX_STANDARD 11)
message(STATUS "C++ version ${CXX_STANDARD} configured.")

# Setup flag overrides
list( APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXX_FLAGS}")

# Create the documentation directory and set up the doc target.
set(DOC_DIR "docs")
if(NOT EXISTS ${DOC_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${DOC_DIR}")
endif(NOT EXISTS ${DOC_DIR})
set(DOXYFILE_OUTPUT_DIR "${DOC_DIR}")
set(DOXYFILE_LATEX "ON")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
include(UseDoxygen OPTIONAL)

# Include directories for the main program. Note that include 
# directories must come before add subdirectory!
include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories("${CMAKE_SOURCE_DIR}/tpls/simpleini-4.17")
include_directories("${CMAKE_SOURCE_DIR}/tpls/fire/integrators")
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/tpls/fire/parsers")

# Add the source directory
add_subdirectory(src)
# Add the third party libraries
ADD_SUBDIRECTORY(tpls)

# Setup the library list
set(FERN_LIBS fern)

# Add an executable for the main program
ADD_EXECUTABLE(fern-exec ${CMAKE_SOURCE_DIR}/src/main.cpp)

# Link the FERN library
target_link_libraries(fern-exec ${FERN_LIBS})

# ---- Tests! ----- #

# Add integration and system tests. The first one just makes sure it executes.
add_test(smoke_test_alpha fern-exec ${CMAKE_SOURCE_DIR}/data/alpha.ini)
# Check the results for the alpha network benchmark
add_test(diff_test_alpha ${CMAKE_SOURCE_DIR}/tools/diffTest.sh ${CMAKE_SOURCE_DIR}/data/alpha_gold.ini ${CMAKE_SOURCE_DIR}/data/alpha_gold.txt)

# Find Boost for unit tests
FIND_PACKAGE(Boost)
# Enable testing if boost is found.
if(Boost_FOUND)
    # Turn on the testing harness
    enable_testing()
    # Let the user know that tests will be built
    message(STATUS "Boost version ${Boost_VERSION} found.")
    message(STATUS "Tests will be built since Boost was found.")
    message(STATUS "Tests can be run with the 'make test' command.")
    # Include the header directory since BOOST tests
    INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIR}")
    #Get the test files
    file(GLOB tests src/*Test.cpp)
    #Make executables and link libraries for testers
    foreach(test ${tests})
        message(STATUS "Making test ${test}")
        get_filename_component(testName ${test} NAME_WE)
        add_executable(${testName} ${test})
	# Link the FERN library
	target_link_libraries(${testName} ${FERN_LIBS})
        # Add the test
        add_test(${testName} ${testName}) 
        # Add a label so the tests can be run separately
        set_property(TEST ${testName} PROPERTY LABELS ${PACKAGE_NAME})   
    endforeach(test ${tests})
endif(Boost_FOUND)
