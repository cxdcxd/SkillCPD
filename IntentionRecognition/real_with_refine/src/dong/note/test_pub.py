#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
import numpy as np

def separate_xdx(mydata_xdx):
    mydata_x = []
    mydata_dx = []
    if np.array(mydata_xdx).ndim == 3:
        for i in range(len(mydata_xdx)):
            temp_x = []
            temp_dx = []
            for j in range(len(mydata_xdx[i])):
                temp_x.append([mydata_xdx[i][j][0], mydata_xdx[i][j][1]])
                temp_dx.append([mydata_xdx[i][j][2], mydata_xdx[i][j][3]])
            temp_x1 = np.array(temp_x)
            temp_dx1 = np.array(temp_dx)
            mydata_x.append(temp_x1)
            mydata_dx.append(temp_dx1)
        return mydata_x, mydata_dx
    elif np.array(mydata_xdx).ndim == 2:
        temp_x = []
        temp_dx = []
        for i in range(len(mydata_xdx)):
            temp_x.append([mydata_xdx[i][0], mydata_xdx[i][1]])
            temp_dx.append([mydata_xdx[i][2], mydata_xdx[i][3]])
        return temp_x, temp_dx
    else:
        print("Error! The dimension of data is neither 2 or 3")

def mytest():
    rospy.init_node('test', anonymous=True)
    pub_qos = rospy.Publisher('QoS', Float64, queue_size=1)
    pub_mouse_input = rospy.Publisher('Mouse_input', Pose, queue_size=1)
    rate = rospy.Rate(1) # 10hz
    mouse_points = Pose()

    data = np.load("/home/test/ma-yang/real_with_refine/src/note/mydataset/demoAhalf.npy", allow_pickle=True)
    data = data.tolist()
    data_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(data['x'], data['dx'])]
    data_x, data_dx = separate_xdx(data_xdx)
    for i in range(len(data_dx[0])):
        # qos = np.random.randint(100)
        # qos = 100
        qos = [100]*100 + [0]*120000
        print('The random QoS is publishing...')
        # pub_qos.publish(qos)
        pub_qos.publish(qos[i])

        mouse_points.position.x = data_x[0][i][0]
        mouse_points.position.y = data_x[0][i][1]
        mouse_points.position.z = 0
        print('The mouse input is publishing...')
        pub_mouse_input.publish(mouse_points)
        rate.sleep()

    # while not rospy.is_shutdown():
    #     qos = np.random.randint(100)
    #     print('The random QoS is publishing...')
    #     pub_qos.publish(qos)
    #     rate.sleep()

if __name__ == '__main__':
    try:
        mytest()
    except rospy.ROSInterruptException:
        pass