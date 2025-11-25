cmake_minimum_required(VERSION 3.8)
project(cuda_utils)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#set(Thrust_DIR "/usr/local/cuda/targets/x86_64-linux/lib/cmake/thrust")
enable_language("CUDA")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(CUDA REQUIRED )

#find_package(Thrust REQUIRED CONFIG)
#thrust_create_target(Thrust)

include_directories(include)

#cuda_include_directories(
#  include
#)


add_library( cuda_utils SHARED
    src/CachedAllocator.cpp
)
target_include_directories(cuda_utils
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(cuda_utils ${CUDA_LIBRARIES})


ament_export_include_directories(include)
#ament_export_libraries(export_cuda_utils)
ament_export_targets(cuda_utils HAS_LIBRARY_TARGET)

install(
  DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
)

install(TARGETS cuda_utils
  EXPORT cuda_utils
  LIBRARY DESTINATION lib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
