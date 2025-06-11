cmake_minimum_required(VERSION 3.5)
project(tensorrt_ros)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

enable_language(CUDA)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(interfaces REQUIRED)

include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)
link_directories(/usr/local/cuda/lib64)


add_executable(tensorrt
  src/tensorrt.cpp  # 替换为您的实际源文件路径
  src/include/postprocess.cpp       # 添加 postprocess 源文件
  src/include/model.cpp
  src/include/preprocess.cu
  src/include/yololayer.cu
  src/include/trt.cpp
  #src/cuda_utils.cu         # 添加 CUDA 源文件
)

link_directories(
  "/usr/local/cuda/lib64"
  "/usr"   # tensorrt 位置
  "./libmyplugins.so"
  )

target_link_libraries(tensorrt
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${vision_msgs_LIBRARIES}
  ${OpenCV_LIBRARIES}  # 添加OpenCV库链接
  ${CUDA_LIBRARIES}
  realsense2::realsense2
  ${TensorRT_LIBRARIES}
  ${CUDA_LIBRARIES}
  cudart
  nvinfer
)

target_include_directories(tensorrt
  PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}
  ${realsense2_INCLUDE_DIRS}  # 添加librealsense2的头文件路径
  ${TensorRT_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${interfaces_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include/tensorrt_ros
)


ament_target_dependencies(tensorrt
  rclcpp
  std_msgs
  sensor_msgs
  interfaces
  OpenCV
)

install(TARGETS tensorrt
  DESTINATION lib/${PROJECT_NAME}
)


ament_export_dependencies(rclcpp)
ament_export_dependencies(std_msgs)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(vision_msgs)
ament_export_dependencies(realsense2)
ament_export_dependencies(OpenCV)
ament_export_dependencies(TensorRT)
ament_export_dependencies(interfaces)

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
