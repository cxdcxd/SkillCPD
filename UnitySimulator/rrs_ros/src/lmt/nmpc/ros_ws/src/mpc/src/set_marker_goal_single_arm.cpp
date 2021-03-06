/*
 * Copyright (c) 2011, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


// %Tag(fullSource)%
#include <ros/ros.h>

#include <interactive_markers/interactive_marker_server.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <chrono>

void processFeedback(
    const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback )
{
    static double goal_pos[3];
    static double goal_quat[4];
    goal_pos[0] = feedback->pose.position.x;
    goal_pos[1] = feedback->pose.position.y;
    goal_pos[2] = feedback->pose.position.z;
    goal_quat[0] = feedback->pose.orientation.x;
    goal_quat[1] = feedback->pose.orientation.y;
    goal_quat[2] = feedback->pose.orientation.z;
    goal_quat[3] = feedback->pose.orientation.w;


    std::cout << "position: " << goal_pos[0] << " " << goal_pos[1] << " " << goal_pos[2] << std::endl;
    std::cout << "quaternion: " << goal_quat[3] << " " << goal_quat[0] << " " << goal_quat[1] << " " << goal_quat[2] << std::endl;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "simple_marker");
  //ros::NodeHandle nh;

  // create an interactive marker server on the topic namespace simple_marker
  interactive_markers::InteractiveMarkerServer server("simple_marker");

  // create an interactive marker for our server
  visualization_msgs::InteractiveMarker int_marker;
  int_marker.header.frame_id = "world";
  int_marker.header.stamp=ros::Time::now();
  int_marker.name = "my_marker";
  int_marker.description = "j2s7s300 eef Control";
  int_marker.scale = 0.2;

  // end effector start position for real robot
   tf2_ros::Buffer tfBuffer;
   tf2_ros::TransformListener tfListener(tfBuffer);
   ROS_INFO("Wait 0.5 second for tf coming in");
   geometry_msgs::TransformStamped transformStamped;
   ros::Duration(0.5).sleep();
   try{
   transformStamped = tfBuffer.lookupTransform("world", /*"j2s7s300_end_effector"*/ "j2s7s300_end_effector",
                            ros::Time(0));
   }
   catch (tf2::TransformException &ex) {
   ROS_WARN("%s",ex.what());
   return 0;
   }




   int_marker.pose.position.x = transformStamped.transform.translation.x;
   int_marker.pose.position.y = transformStamped.transform.translation.y;;
   int_marker.pose.position.z = transformStamped.transform.translation.z;;

   int_marker.pose.orientation.x = transformStamped.transform.rotation.x;
   int_marker.pose.orientation.y = transformStamped.transform.rotation.y;
   int_marker.pose.orientation.z = transformStamped.transform.rotation.z;
   int_marker.pose.orientation.w = transformStamped.transform.rotation.w;

//    this int marker controls obstacle pose
//    int_marker.pose.position.x = 0;
//    int_marker.pose.position.y = 1.0;
//    int_marker.pose.position.z = 1.0;

//    int_marker.pose.orientation.x = 0;
//    int_marker.pose.orientation.y = 0;
//    int_marker.pose.orientation.z = 0;
//    int_marker.pose.orientation.w = 1;


  // create a grey box marker
  visualization_msgs::Marker box_marker;
  box_marker.type = visualization_msgs::Marker::CUBE;
  box_marker.scale.x = 0.045;
  box_marker.scale.y = 0.045;
  box_marker.scale.z = 0.045;
  box_marker.color.r = 0.5;
  box_marker.color.g = 0.5;
  box_marker.color.b = 0.5;
  box_marker.color.a = 1.0;




  // create a non-interactive control which contains the box
  visualization_msgs::InteractiveMarkerControl box_control;
  box_control.always_visible = true;
  box_control.markers.push_back( box_marker );

  // add the control to the interactive marker
  int_marker.controls.push_back( box_control );

  // create a control which will move the box
  // this control does not contain any markers,
  // which will cause RViz to insert two arrows
  visualization_msgs::InteractiveMarkerControl control;
  /*
  rotate_control.name = "move_x";
  rotate_control.interaction_mode =
      visualization_msgs::InteractiveMarkerControl::MOVE_3D;
  */

  tf::Quaternion orien(1.0, 0.0, 0.0, 1.0);
  orien.normalize();
  tf::quaternionTFToMsg(orien, control.orientation);
  control.name = "rotate_x";
  control.interaction_mode = visualization_msgs::InteractiveMarkerControl::ROTATE_AXIS;
  int_marker.controls.push_back(control);
  control.name = "move_x";
  control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
  int_marker.controls.push_back(control);

  orien = tf::Quaternion(0.0, 1.0, 0.0, 1.0);
  orien.normalize();
  tf::quaternionTFToMsg(orien, control.orientation);
  control.name = "rotate_z";
  control.interaction_mode = visualization_msgs::InteractiveMarkerControl::ROTATE_AXIS;
  int_marker.controls.push_back(control);
  control.name = "move_z";
  control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
  int_marker.controls.push_back(control);

  orien = tf::Quaternion(0.0, 0.0, 1.0, 1.0);
  orien.normalize();
  tf::quaternionTFToMsg(orien, control.orientation);
  control.name = "rotate_y";
  control.interaction_mode = visualization_msgs::InteractiveMarkerControl::ROTATE_AXIS;
  int_marker.controls.push_back(control);
  control.name = "move_y";
  control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
  int_marker.controls.push_back(control);

  // add the control to the interactive marker
  //int_marker.controls.push_back(control);

  // add the interactive marker to our collection &
  // tell the server to call processFeedback() when feedback arrives for it
  server.insert(int_marker, &processFeedback);

  // 'commit' changes and send to all clients
  server.applyChanges();

  // start the ROS main loop
  ros::spin();
}
// %Tag(fullSource)%
