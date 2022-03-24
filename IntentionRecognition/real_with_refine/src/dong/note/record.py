#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
import numpy as np

result_pose = []
result_ori = []
trajectory_points = Pose()
finish_pub = 0
sample = 0

def record_3d():
    global trajectory_points, result_pose, result_ori, finish_pub, sample
    rospy.init_node('record', anonymous=True)
    rospy.Subscriber('right/nmpc_controller/in/goal', Pose, callback_record)
    while 1:
        rospy.sleep(0.1)
        if finish_pub > 40 * 5:
            np.save('record_yellow_4.npy', [result_pose, result_ori])
            print([result_pose, result_ori])
            break

def callback_record(msg_record):
    global trajectory_points, result_pose, result_ori, finish_pub, sample
    # trajectory_points = Pose()
    # Hz = 60
    sample += 1
    if sample >= 12:
        sample = 0
        trajectory_points = msg_record
        result_pose += [[trajectory_points.position.x, trajectory_points.position.y, trajectory_points.position.z]]
        result_ori += [[trajectory_points.orientation.x, trajectory_points.orientation.y, trajectory_points.orientation.z, trajectory_points.orientation.w]]
        finish_pub += 1
    # print('One point is recorded...')

if __name__ == '__main__':
    try:
        record_3d()
    except rospy.ROSInterruptException:
        pass