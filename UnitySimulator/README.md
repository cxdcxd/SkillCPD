# Realistic Robotic Simulator (RRS)

| OS  | Kernel Version | ROS Version | Nvidia Driver Version | CUDA Version | Unity Version | Ml Agents
| --- | ----------| ----------- | ------------ | ------------ | ------------ | ------------ 
| Ubuntu 18.04.06 LTS | 5.4.0-58-generic | ROS Melodic | 460.27.04 | 11.2 | Unity 2020.3.22f1 | 1.0.0

<!--# Unity Version
    Unity 2020.3.22f1-->
    
<!--# ROS Version
    Ubuntu 18.04.06
    ROS Melodic-->

# Setup and Installation

1) Install [ROS](http://wiki.ros.org/melodic/Installation/Ubuntu)

2) Clone this Repository and run the automatic installer
```
git clone https://github.com/cxdcxd/RRS.git
cd rrs_ros
./install.sh
```

3) Download [Unity](https://unity3d.com/get-unity/download/archive) 2020.2.1f1 or above

4) Open Unity Hub and go to `Projects`, click `ADD` and then browse the folder `rrs_unity` from the downloaded repository and launch it. Make sure the `Unity Version` is set to `2020.3.22f1` or above

5) Open the repository in a new terminal and type `cd rrs_ros` and build the workspace: `catkin_make`

6) Source the workspace: 
```
setup ~/rrs_ros/devel/setup.bash
```

# Run examples

## ROS Side:
Benchmark scene
```
roslaunch rrs_ros rrs_ros_benchmark.launch                          //for benchmark scene
```
Movo Robot Simulator scene
```
roslaunch rrs_ros rrs_ros.launch                                    //for core robot simulation
roslaunch rrs_ros rrs_moveit.launch                                 //for moveit 
roslaunch jog_launch jaco2.launch use_moveit:=true use_rviz:=true   //for arm jogging 
```

## Unity Side:
Benchmark scene
```
Open the scenes/BenchMark
Unity3D -> Play
```
Movo Robot Simulator
```
open the scenes/Demo
Unity3D -> Play
```

# Settings (rrs_ros) 
### open the config file from : (RRS/rrs_ros/src/core/cfg/config.yaml) 
    ntp_server_host_name: test               //define the ntp server hostname (current RRS is not using ntp ignored)
    local_network_address: 127.0.0.1         //local ip address
    consul_network_address: 127.0.0.1        //consul ip address
    consul_network_mask: 255.255.255.0       //consul network mask
    consul_network_port: 8500                //consul network port
    
# Settings (rrs_unity) 
### open the config file from : (RRS/rrs_unity/Config/config.xml)
    <?xml version="1.0" encoding="utf-8"?>
    <Config xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
      <consul_network_address>127.0.0.1</consul_network_address>
      <local_network_address>127.0.0.1</local_network_address>
      <consul_network_port>8500</consul_network_port>
      <consul_network_mask>255.255.255.0</consul_network_mask>
      <ntp_server_host_name>test</ntp_server_host_name> //define the ntp server hostname (current RRS is not using ntp ignored)
      <use_relative_origin>false</use_relative_origin>
    </Config>
    
