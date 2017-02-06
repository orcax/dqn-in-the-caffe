cmake_minimum_required (VERSION 2.8.9)
project(dqn)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

option(CPU_ONLY "Use CPU only for Caffe" OFF)
option(USE_CUDNN "Use cuDNN for Caffe" OFF)
option(USE_SDL "Use SDL for ALE" OFF)

link_directories(./caffe/build/lib)
include_directories(./caffe/include)

set(XITARI_LIB libxitari.so)
link_directories(./xitari)
include_directories(./xitari ./xitari/common)

if(USE_CUDNN)
  link_directories(~/cudnn-6.5-linux-R1)
endif()

add_executable(dqn dqn_main.cpp dqn.cpp)

target_link_libraries(dqn caffe)
target_link_libraries(dqn glog)
target_link_libraries(dqn gflags)
target_link_libraries(dqn ${XITARI_LIB})

if(APPLE)
  include_directories(/System/Library/Frameworks/vecLib.framework/Versions/Current/Headers)
endif()

if(USE_SDL)
  add_definitions(-D__USE_SDL)
  if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Cocoa")
    target_link_libraries(dqn sdl)
    target_link_libraries(dqn sdl_gfx)
    target_link_libraries(dqn sdl_image)
    target_link_libraries(dqn sdlmain)
  else()
    target_link_libraries(dqn SDL)
    target_link_libraries(dqn SDL_gfx)
    target_link_libraries(dqn SDL_image)
    target_link_libraries(dqn SDLmain)
  endif()
endif()

if(NOT CPU_ONLY)
  include_directories(/usr/local/cuda/targets/x86_64-linux/include)
endif()

if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()

if(USE_CUDNN)
  target_link_libraries(dqn cudnn)
endif()

