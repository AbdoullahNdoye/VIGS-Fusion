cmake_minimum_required(VERSION 3.8)
project(vigs_fusion)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language("CUDA")

set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "-std=c++17")
set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
set(CUDA_UTILS_DIR ${PROJECT_SOURCE_DIR}/include/cuda_utils/build)

list(APPEND CMAKE_CUDA_FLAGS "-w")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cuda_utils REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(CUDA REQUIRED )
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  include
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include/cuda_utils
)

add_library(GaussianSplattingSlam SHARED
            src/GaussianSplattingSlam.cu
            src/GaussianSplattingSlamKernels.cu
            src/GaussianSplattingSlamTypes.cu
            src/GaussianSplattingKeyframe.cu
            src/Preintegration.cpp
            src/Utility.cpp
            src/PoseLocalParameterization.cpp
            src/GSPoseCost.cu
            src/ImuCostFunction.cpp
            src/MarginalizationFactor.cpp
            src/BetaBinomialGenerator.cpp
            src/GaussianSplattingViewer.cpp
)

ament_target_dependencies( GaussianSplattingSlam
  cuda_utils
)

target_include_directories(GaussianSplattingSlam
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  include/cuda_utils/include
)

target_link_libraries(GaussianSplattingSlam
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CERES_LIBRARIES}
  ${CUDA_UTILS_DIR}/libcuda_utils.so

)
set_target_properties(GaussianSplattingSlam
  PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

add_library(GaussianSplattingSlamNode SHARED
            src/GaussianSplattingSlamNode.cpp
)

ament_target_dependencies( GaussianSplattingSlamNode
  rclcpp
  cuda_utils
  nav_msgs
  image_transport
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  image_geometry   
  message_filters
  sensor_msgs
  cv_bridge
)
target_include_directories(GaussianSplattingSlamNode
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(GaussianSplattingSlamNode
  GaussianSplattingSlam
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  ${CUDA_UTILS_DIR}/libcuda_utils.so
)


add_executable(vigs_fusion
              src/gaussian_splatting_slam
)
target_link_libraries(vigs_fusion
  GaussianSplattingSlam
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CERES_LIBRARIES}
)

add_executable(gaussian_splatting_slam_node
  src/gaussian_splatting_slam_node.cpp
)

target_link_libraries(gaussian_splatting_slam_node
  GaussianSplattingSlamNode
)

install(TARGETS 
  GaussianSplattingSlam
  GaussianSplattingSlamNode
  vigs_fusion
  gaussian_splatting_slam_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

install(DIRECTORY
	launch
	DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
