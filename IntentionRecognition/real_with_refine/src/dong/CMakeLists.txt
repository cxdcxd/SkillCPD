# toplevel CMakeLists.txt for a catkin workspace
# catkin/cmake/toplevel.cmake

cmake_minimum_required(VERSION 3.0.2)

project(dong_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  joint_control
)

catkin_package(
CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs joint_control
)

catkin_python_setup()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS note/online_record.py 
                      note/test_pub.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})