#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
import numpy as np


def network_qos():
    rospy.init_node('network_parameter', anonymous=True)
    pub_qos = rospy.Publisher('QoS', Float64, queue_size=1)
    rate = rospy.Rate(1)
    qos = [1]
    # qos = [100]*100 + [0]*1200
    # for i in range(9):
    #     qos += [np.random.randint(2)]

    # QoS is random
    for i in range(9999):
        if i % 2 == 0:
            qos += [np.random.randint(2)]
        else:
            qos += [1]
        print('The random QoS is publishing...')
        pub_qos.publish(qos[i])
        rate.sleep()

    # QoS is always 1 for direct teleoperation
    qos = 1

    # while not rospy.is_shutdown():
    #     qos = np.random.randint(2)
    #     pub_qos.publish(qos)
    #     print('The random QoS is publishing...')
    #     rate.sleep()

if __name__ == '__main__':
    try:
        network_qos()
    except rospy.ROSInterruptException:
        pass