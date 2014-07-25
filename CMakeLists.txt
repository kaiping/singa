cmake_minimum_required (VERSION 2.6) 
cmake_policy(SET CMP0014 OLD)

project (LAPIS) 

#set(ENV{LANG} "C")

find_package(SWIG REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(MPI REQUIRED)
find_package( Boost COMPONENTS system thread REQUIRED )
include(FindProtobuf)
find_package(Protobuf REQUIRED)


set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

set(CMAKE_CXX_FLAGS "-fPIC -march=core2 -ggdb2 -Wall -Wno-sign-compare -Wno-unused-function -fno-omit-frame-pointer -std=c++11")
if ($ENV{TCMALLOC})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHEAPPROF ")
endif()

if ($ENV{OPROFILE})
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPUPROF ")
endif()

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory (external/google-test)
add_subdirectory (external/google-flags)
add_subdirectory (external/google-logging)

include_directories ("include")
include_directories (${Boost_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIRS})
include_directories (external/google-test)
include_directories (external/google-flags)
include_directories (external/google-logging)


exec_program("mpic++ -showme:compile" OUTPUT_VARIABLE MPI_COMPILE_FLAGS)
exec_program("mpic++ -showme:incdirs" OUTPUT_VARIABLE MPI_INCDIRS)
exec_program("mpic++ -showme:link" OUTPUT_VARIABLE MPI_LINK_FLAGS)
exec_program("mpic++ -showme:libdirs" OUTPUT_VARIABLE MPI_LIBDIRS)

string(REPLACE " " ";" MPI_INCDIRS "${MPI_INCDIRS}")
string(REPLACE " " ";" MPI_LINK_FLAGS "${MPI_LINK_FLAGS}")

include_directories(${CMAKE_BINARY_DIR}/src
                     ${MPI_INCDIRS} )

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

link_directories( ${MPI_LIBDIRS} )

add_subdirectory (src/proto) 
add_subdirectory (src/utils) # compile GlobalContext
add_subdirectory (src/core) # compile core
#add_subdirectory (src/core)
