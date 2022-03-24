#!/usr/bin/env python
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
import numpy as np


res = np.load("record_orange_0.npy", allow_pickle=True)

print(len(res[0]))

print(res[0][294])
# print(res[1][1])