#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
import numpy as np


def check_record_3d():
    record_points = np.load("record_red_4.npy", allow_pickle=True)
    result_pub = Pose()
    rospy.init_node('test', anonymous=True)
    pub_trajectoy = rospy.Publisher('right/nmpc_controller/in/goal', Pose, queue_size=1)
    res = Pose()
    rate = rospy.Rate(5) # 5hz

    for i in range(len(record_points[0])):
        
        result_pub.position.x = record_points[0][i][0]
        result_pub.position.y = record_points[0][i][1]
        result_pub.position.z = record_points[0][i][2]
        result_pub.orientation.x = record_points[1][i][0]
        result_pub.orientation.y = record_points[1][i][1]
        result_pub.orientation.z = record_points[1][i][2]
        result_pub.orientation.w = record_points[1][i][3]
        pub_trajectoy.publish(result_pub)
        print(result_pub.position.x)
        print('One point is published...')
        rate.sleep()

if __name__ == '__main__':
    try:
        check_record_3d()
    except rospy.ROSInterruptException:
        pass