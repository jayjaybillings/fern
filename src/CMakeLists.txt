#Set the package name
SET(PACKAGE_NAME "fern")
#Set the description
SET(PACKAGE_DESCRIPTION "FERN Core Library")
#Set the library name
SET(LIBRARY_NAME "fern")

#Collect all header filenames in this project 
#and glob them in HEADERS
file(GLOB HEADERS *.hpp)

#Grab all of the source files
file(GLOB SRC *.cpp)

#Include the integrator director for time steppers
include_directories("${CMAKE_SOURCE_DIR}/tpls/fire/integrators")

#Add the source code to the library
add_library(${LIBRARY_NAME} STATIC ${SRC})
#target_link_libraries(${LIBRARY_NAME} fireIntegrators)

#Install the FERN header files
install(FILES ${HEADERS} DESTINATION include)
