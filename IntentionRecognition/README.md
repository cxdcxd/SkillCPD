roscore

start the gazebo environment:

    $ ~/movo_ws: source devel/setup.bash

    $ ~/movo_ws: roslaunch movo_demos sim_demo_show_basic.launch

start siqi's system:

    $ ~/siqi/ma-siqi/ros_ws: source devel setup.bash

    $ ~/siqi/ma-siqi/ros_ws: roslaunch mpc sim_movo_sc.launch

In a new terminal: 

    $ rosservice call /right/j2s7s300/track_toggl"{}"

To start the demo of a example for letter A:

    $ ~/ma-yang/real_with_refine: source devel setup.bash

    $ ~/ma-yang/real_with_refine: rosrun dong_pkg online_record.py 

In a new terminal: 

    $ ~/ma-yang/real_with_refine: source devel setup.bash

    $ ~/ma-yang/real_with_refine: rosrun dong_pkg trajectory_pub.py 

In a new terminal:

    $ ~/ma-yang/real_with_refine: source devel setup.bash

    $ ~/ma-yang/real_with_refine: rosrun dong_pkg test_pub.py 
