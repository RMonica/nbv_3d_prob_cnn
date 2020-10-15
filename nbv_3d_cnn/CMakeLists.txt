cmake_minimum_required(VERSION 2.8.3)
project(nbv_3d_cnn)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  actionlib
  sensor_msgs
  actionlib_msgs
  visualization_msgs
  eigen_conversions
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)

find_package(OpenCL REQUIRED)
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=110)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)

add_definitions(-Wno-ignored-attributes)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   Floats.msg
 )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

add_action_files(
  FILES
  Predict.action
  Predict3d.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  actionlib_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES nbv_3d_cnn
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

add_custom_command(
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_LIST_DIR}/cmake/generate_test_dataset_cl.cmake"
  MAIN_DEPENDENCY "${CMAKE_CURRENT_LIST_DIR}/src/generate_test_dataset.cl"
  OUTPUT "${CMAKE_CURRENT_LIST_DIR}/src/generate_test_dataset.cl.h"
  VERBATIM
  )

add_executable(generate_test_dataset
  src/generate_test_dataset.cpp
  src/generate_test_dataset_opencl.cpp
  src/generate_test_dataset.cl.h
  src/origin_visibility.cpp
  src/origin_visibility_ocl.cpp
  src/generate_single_image.cpp
  src/voxelgrid.cpp
)

add_dependencies(generate_test_dataset ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(generate_test_dataset
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(augment_test_dataset
  src/augment_test_dataset.cpp
  src/generate_test_dataset_opencl.cpp
  src/generate_test_dataset.cl.h
  src/origin_visibility.cpp
  src/origin_visibility_ocl.cpp
  src/voxelgrid.cpp
)

add_dependencies(augment_test_dataset ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(augment_test_dataset
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(simulate_nbv_cycle
  src/simulate_nbv_cycle.cpp
  src/simulate_nbv_cycle_adapter.cpp
  src/generate_test_dataset_opencl.cpp
  src/generate_test_dataset.cl.h
  src/origin_visibility.cpp
  src/origin_visibility_ocl.cpp
  src/generate_single_image.cpp
  src/voxelgrid.cpp
)

add_dependencies(simulate_nbv_cycle ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(simulate_nbv_cycle
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(test_visibility_matrix
  src/test_visibility_matrix.cpp
  src/generate_test_dataset_opencl.cpp
  src/generate_test_dataset.cl.h
  src/origin_visibility.cpp
  src/origin_visibility_ocl.cpp
  src/generate_single_image.cpp
  src/voxelgrid.cpp
)

add_dependencies(test_visibility_matrix ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(test_visibility_matrix
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(voxelgrid_visualization
  src/voxelgrid_visualization.cpp
  src/voxelgrid.cpp
)

add_dependencies(voxelgrid_visualization ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(voxelgrid_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(merge_logs
  src/merge_logs.cpp
)

add_dependencies(merge_logs ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(merge_logs
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

add_executable(voxelgrid_to_octomap
  src/voxelgrid_to_octomap.cpp
  src/voxelgrid.cpp
)

add_dependencies(voxelgrid_to_octomap ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(voxelgrid_to_octomap
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(voxelgrid_to_binvoxelgrid
  src/voxelgrid_to_binvoxelgrid.cpp
  src/voxelgrid.cpp
)

add_dependencies(voxelgrid_to_binvoxelgrid ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(voxelgrid_to_binvoxelgrid
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_executable(binvoxelgrid_to_voxelgrid
  src/binvoxelgrid_to_voxelgrid.cpp
  src/voxelgrid.cpp
)

add_dependencies(binvoxelgrid_to_voxelgrid ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(binvoxelgrid_to_voxelgrid
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

include_directories(
  ${OCTOMAP_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(generate_3d_scenes src/generate_3d_scenes.cpp)

add_dependencies(generate_3d_scenes ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(generate_3d_scenes
  ${catkin_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_nbv_3d_cnn.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
