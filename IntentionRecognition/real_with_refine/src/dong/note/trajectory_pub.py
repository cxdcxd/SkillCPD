#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
import numpy as np

trajectory_points = Pose()
result_pose = []
result_ori = []
finish_pub = 0

def trajectory_sub_pub():
    global trajectory_points, result_pose, result_ori
    result_pub = Pose()
    rospy.init_node('test', anonymous=True)
    pub_trajectoy = rospy.Publisher('right/nmpc_controller/in/goal', Pose, queue_size=1)
    rospy.Subscriber('finish_pub', Float64, callback_finish)
    rospy.Subscriber('trajectory_points', Pose, callback_trajectory)
    print("Waiting for reaching the start point of the trajectory...")
    rospy.sleep(80)
    while 1:
        if finish_pub:
            print("Start to publish the remaining points...")
            rate = rospy.Rate(2) # 2hz
            for i in range(len(result_pose)):
                result_pub.position.x = result_pose[i][0]
                result_pub.position.y = result_pose[i][1]
                result_pub.position.z = result_pose[i][2]
                result_pub.orientation.x = result_ori[0]
                result_pub.orientation.y = result_ori[1]
                result_pub.orientation.z = result_ori[2]
                result_pub.orientation.w = result_ori[3]
                pub_trajectoy.publish(result_pub)
                rate.sleep()
            print('The trajectory consists of %d points' %(i+1))
            break
        else:
            continue

def callback_finish(msg_finish):
    global finish_pub
    finish_pub = msg_finish.data

def callback_trajectory(msg_pose):
    global trajectory_points, result_pose, result_ori
    trajectory_points = msg_pose
    result_pose += [[trajectory_points.position.x, trajectory_points.position.y, trajectory_points.position.z]]
    result_ori = [trajectory_points.orientation.x, trajectory_points.orientation.y, trajectory_points.orientation.z, trajectory_points.orientation.w]

if __name__ == '__main__':
    try:
        trajectory_sub_pub()
    except rospy.ROSInterruptException:
        pass