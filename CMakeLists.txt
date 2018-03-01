cmake_minimum_required(VERSION 2.8.3)
project(cibr_img_processing)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries

find_package(catkin REQUIRED COMPONENTS message_generation
  rospy
  sensor_msgs
#  reflex_msgs
  cv_bridge
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Uncomment this if the package has a setup.py. # catkin_python_setup()


## message message_generation

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   Ints.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES cibr_img_processing
  CATKIN_DEPENDS message_runtime rospy sensor_msgs cv_bridge # reflex_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
