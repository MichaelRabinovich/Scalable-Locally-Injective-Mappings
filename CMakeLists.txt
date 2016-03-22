cmake_minimum_required(VERSION 2.8)
project(ReweightedARAP)

set(CMAKE_COLOR_MAKEFILE ON)
include(CheckCXXCompilerFlag)

# Set a default build configuration if none is specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)

  if (HAS_CPP11_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(FATAL_ERROR "Unsupported compiler -- ReweightedARAP requires C++11 support!")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -Wno-deprecated-declarations")
endif()

if (MSVC)
  # Enable parallel compilation for Visual Studio
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj") 

  # Enable folders for projects in Visual Studio
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

set(CMAKE_MACOSX_RPATH 1)

# Check to see if PARDISO is available
find_library(PARDISO
  pardiso500-INTEL1301-X86-64
  pardiso500-GNU461-X86-64
  pardiso500-GNU472-X86-64
  pardiso500-GNU481-X86-64
  pardiso500-MPI-INTEL1301-X86-64
  pardiso500-MPI-GNU450-X86-64
  pardiso500-MPI-GNU461-X86-64
  pardiso500-MPI-GNU463-X86-64
  pardiso500-MPI-GNU472-X86-64
  pardiso500-WIN-X86-64
  pardiso500-MACOS-X86-64
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ext/pardiso
)

set(ADDITIONAL_LIBRARIES "")

if("${CMAKE_SYSTEM}" MATCHES "Linux")
  set(ADDITIONAL_LIBRARIES rt pthread dl)
endif()

if (${PARDISO} MATCHES "NOTFOUND")
  message("\n*NOT* using PARDISO -- note that the performance will be considerably slower than what is shown in the paper.\n")
else()
  message("Using PARDISO library at ${PARDISO}")
  add_definitions(-DUSE_PARDISO)
  set(ADDITIONAL_LIBRARIES ${PARDISO})
  set(ADDITIONAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PardisoSolver.cpp
  )
endif()

# Compile thread building blocks
set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/tbb ext_build/tbb)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/libigl/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/tbb/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen
)

set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/eigen_stl_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/parametrization_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Param_State.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/LocalWeightedArapParametrizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GlobalLocalParametrization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/LinesearchParametrizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/SymmetricDirichlet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/FastLsBuildUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/StateManager.cpp
  ${ADDITIONAL_SOURCES}
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} tbb_static ${ADDITIONAL_LIBRARIES})
