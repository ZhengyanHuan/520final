import sys
import numpy as np
from copy import deepcopy
from math import pi
from lib.Vec2H import transform, select_rot
import solveIK

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds




if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    # get the transform from camera to panda_end_effector

    H_ee_camera = detector.get_H_ee_camera()
    # print(H_ee_camera)

    # Detect some blocks...
    startH = np.array([[ 1.,     0.,    -0.,     0.562]
    , [ 0.,    -1.,     0.,    -0.169]
    ,[-0.,    -0.,    -1.,     0.538]
    ,[ 0.,    0.,     0.,    1.   ]])

    start_position2 = np.array([-0.14589, 0.1306, -0.16275, -1.36351, 0.02117, 1.49242, 0.47977])
    #    target = transform(np.array([0.562,-0.169,0.538]), np.array([0,pi,pi]))
    arm.safe_move_to_position(start_position2)
    arm.exec_gripper_cmd(0.2, 50)

    cubeH_list = []
    for (name, pose) in detector.get_detections():
         print(name,'\n',pose)
         cubeH_list.append(pose)


    prepos = start_position2
    ik = solveIK.IK()
    height = 0
    for i in range(4):
        H1 = startH@H_ee_camera@cubeH_list[i]
        print(H1)
        H1 = H1@select_rot(H1, 0.999)
        ###############################
        Htmp = H1.copy()
        Htmp[2][3] += (0.1+height)
        # print(H1)
        # print(Htmp)
        # # print(H1)
        q, success, rollout = ik.inverse(Htmp, start_position2)
        tmppos = rollout[-1]
        arm.safe_move_to_position(tmppos)
        #
        #####################################
        q, success, rollout = ik.inverse(H1, prepos)
        prepos = rollout[-1]
        # prepos = pos

        arm.safe_move_to_position(prepos)
        arm.exec_gripper_cmd(0.02, 50)

        # Htmp = H1.copy()
        # Htmp[2][3] += (0.1+height)
        # q, success, rollout = ik.inverse(Htmp, prepos)
        # prepos = rollout[-1]
        arm.safe_move_to_position(tmppos)


        destpos = np.array([0.562, 0.169, 0.25+height])
        destangle = np.array([0,pi,pi])
        destH = transform(destpos, destangle)
        q, success, rollout = ik.inverse(destH, tmppos)
        # destconfig = np.array([ 0.41668, 0.35412,  0.01618, -1.93268, -0.00744,  2.28674,  1.22214])
        posdest = rollout[-1]
        arm.safe_move_to_position(posdest)
        arm.exec_gripper_cmd(0.2, 50)

        destpos = np.array([0.562, 0.169, 0.25 + height +0.1])
        destangle = np.array([0, pi, pi])
        destH = transform(destpos, destangle)
        q, success, rollout = ik.inverse(destH, posdest)
        posdest = rollout[-1]
        arm.safe_move_to_position(posdest)

        height+=0.05

    # Move around...

    # END STUDENT CODE

