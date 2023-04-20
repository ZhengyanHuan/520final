# v1: Preliminary realize stacking static blocks
# v2: 4/15/2023
#       1. Solve the problem that sometimes the gripper cannot grasp the side of a block
#       2. Add the optimization function of make the gripper not rotate too much in the xy-plane
#       3. Raise the height of taking photo of the blocks. This should be changed according to reality.
#       4. Simplify the code of solveIK
#       5. Add some comments
#       6. Possible to optimize: See the comments
# v2.1: 4/15/2023
#       1. improve the select rotation function, do not need threshold any more.
# v2.2: 4/16/2023
#       1. If solveIK fails, retry once. If still fails, give up moving this block
#       2. If the gripper somehow fails to grip the block, trying moving other block directly.
#       These two points are used for handling unforeseen problems in reality. I have not encountered
#       such situation in simulation yet.
# v2.3: 4/16/2023
#       1. The code now works for blue team
# v3:   4/18/2023
#       1. Preliminary realize stacking dynamic blocks
# v3.1: 4/20/2023
#       1. Now the code only need solveIK once for dynamic blocks and twice for static blocks
#       2. Update the prediction function for dynamic blocks
#       3. Solve the problem that when failed to grab the dynamic block, the robot won't return to previous position


import sys
import numpy as np
from copy import deepcopy
from math import pi
from lib.Vec2H import transform, select_rot, opt_pos, predict, opt_pos_D
import solveIK

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

redposdict = {23: np.array([ 0.26157,  0.2257 ,  0.03175, -2.04018, -0.00925,  2.26575,
        1.08384]), 28: np.array([ 0.22682,  0.16267,  0.06805, -1.9887 , -0.01317,  2.15096,
        1.08659]), 33: np.array([ 0.20227,  0.11598,  0.09347, -1.91749, -0.01207,  2.03294,
        1.08589]), 38: np.array([ 0.18746,  0.08666,  0.10867, -1.82577, -0.00996,  1.9119 ,
        1.08445]), 43: np.array([ 0.18232,  0.07598,  0.11436, -1.71169, -0.00887,  1.78717,
        1.08366]), 48: np.array([ 0.1873 ,  0.08603,  0.11088, -1.57128, -0.00954,  1.65678,
        1.08399]), 53: np.array([ 0.20459,  0.12106,  0.09637, -1.39595, -0.01164,  1.51645,
        1.08503]), 58: np.array([ 0.23787,  0.19247,  0.06562, -1.16354, -0.01284,  1.35562,
        1.08495]), 63: np.array([ 0.2903 ,  0.3573 ,  0.00298, -0.76834, -0.00115,  1.12564,
        1.07798]), 25: np.array([ 0.24643,  0.19859,  0.04757, -2.02194, -0.01178,  2.22027,
        1.08559]), 30: np.array([ 0.2158 ,  0.14197,  0.0795 , -1.96261, -0.01305,  2.10411,
        1.08654]), 35: np.array([ 0.19519,  0.1021 ,  0.10073, -1.88334, -0.0112 ,  1.9849 ,
        1.0853 ]), 40: np.array([ 0.18428,  0.08006,  0.112  , -1.78298, -0.00933,  1.86253,
        1.08401]), 45: np.array([ 0.18304,  0.07735,  0.11409, -1.65902, -0.00892,  1.73586,
        1.08366]), 50: np.array([ 0.19258,  0.09666,  0.10657, -1.5061 , -0.01027,  1.60221,
        1.08438]), 55: np.array([ 0.21572,  0.1441 ,  0.08653, -1.31205, -0.01249,  1.45562,
        1.08532]), 60: np.array([ 0.25669,  0.23861,  0.04606, -1.04074, -0.01136,  1.27912,
        1.08358])}

blueposdict = {23: np.array([-0.1161 ,  0.2293 , -0.18308, -2.03992,  0.05386,  2.2648 ,
        0.45644]), 28: np.array([-0.12561,  0.16469, -0.1736 , -1.98861,  0.03386,  2.1506 ,
        0.46994]), 33: np.array([-0.13279,  0.11704, -0.1658 , -1.91746,  0.02153,  2.03281,
        0.47833]), 38: np.array([-0.1374 ,  0.08725, -0.16067, -1.82576,  0.01479,  1.91185,
        0.48298]), 43: np.array([-0.14001,  0.07643, -0.15846, -1.71168,  0.01234,  1.78714,
        0.48473]), 48: np.array([-0.14163,  0.08656, -0.15923, -1.57127,  0.01376,  1.65674,
        0.48394]), 53: np.array([-0.14482,  0.12201, -0.16228, -1.39592,  0.01969,  1.51635,
        0.48056]), 58: np.array([-0.15665,  0.1943 , -0.16411, -1.16344,  0.0323 ,  1.3553 ,
        0.47458]), 63: np.array([-0.20666,  0.35953, -0.1405 , -0.76799,  0.05464,  1.12485,
        0.4707 ]), 25: np.array([-0.12017,  0.20148, -0.17915, -2.02176,  0.04477,  2.21962,
        0.46256]), 30: np.array([-0.12878,  0.14354, -0.17022, -1.96255,  0.02814,  2.10387,
        0.47382]), 35: np.array([-0.13495,  0.10292, -0.16337, -1.88332,  0.01825,  1.98482,
        0.48058]), 40: np.array([-0.13865,  0.08056, -0.15942, -1.78297,  0.01334,  1.86249,
        0.484  ]), 45: np.array([-0.14068,  0.0778 , -0.15843, -1.65901,  0.01243,  1.73583,
        0.48472]), 50: np.array([-0.14248,  0.09732, -0.16026, -1.50609,  0.01551,  1.60216,
        0.48291]), 55: np.array([-0.1478 ,  0.14534, -0.16358, -1.312  ,  0.02375,  1.45546,
        0.47845]), 60: np.array([-0.1681 ,  0.24085, -0.16154, -1.04058,  0.04007,  1.27863,
        0.47189])}



if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array(
        [-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353 + pi / 2, 0.75344866])
    arm.safe_move_to_position(start_position)  # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")  # get set!
    print("Go!\n")  # go!
    height = 0
    ik = solveIK.IK()
    base = 0.25
    # STUDENT CODE HERE

    # get the transform from camera to panda_end_effector

    H_ee_camera = detector.get_H_ee_camera()
    # print(H_ee_camera)


    if team == 'red':
        startH = np.array([[1., 0., -0., 0.562]
                              , [0., -1., 0., -0.169]
                              , [-0., -0., -1., 0.6]
                              , [0., 0., 0., 1.]]) # change this and use this in solveik to get the start position: start_position2

        ydest = 0.169
        start_position2 = np.array([-0.1681, 0.24085, -0.16154, -1.04058, 0.04007, 1.27863, 0.47189])
        start_position_n = np.array([-0.14589, 0.1306, -0.16275, -1.36351, 0.02117, 1.49242, 0.47977])
        posdest = start_position_n
    else:
        startH = np.array([[1., 0., -0., 0.562]
                              , [0., -1., 0., 0.169]
                              , [-0., -0., -1., 0.6]
                              , [0., 0., 0.,1.]])  # change this and use this in solveik to get the start position: start_position2
        ydest = -0.169
        start_position2 = np.array([ 0.25669,  0.23861,  0.04605, -1.04074, -0.01136,  1.27912,  1.08358])
        start_position_n = np.array([-0.14589, 0.1306, -0.16275, -1.36351, 0.02117, 1.49242, 0.47977])
        posdest = start_position_n
    # The position to take a photo of the static blocks, should be changed according to real situaltion
    #    target = transform(np.array([0.562,-0.169,0.538]), np.array([0,pi,pi]))
    arm.safe_move_to_position(start_position2)
    arm.exec_gripper_cmd(0.2, 50)  # open the gripper
    print(arm.get_gripper_state())

    cubeH_list = []
    # Detect some static blocks...
    for (name, pose) in detector.get_detections():
        print(name, '\n', pose)
        cubeH_list.append(pose)



    # base = 0.23
    for i in range(4):
        H1 = startH @ H_ee_camera @ cubeH_list[i]  # calculate the position of a block w.r.t to ROBOT's world frame
        # print("cubeH")
        # print(cubeH_list[i])  # the position of a block w.r.t to camera
        # print("H1")
        # print(H1)  # the position of a block w.r.t to ROBOT's world frame
        # print(select_rot(H1))  # The threshold should be changed according to reality
        H1 = H1 @ select_rot(H1)  # rotate to make z always point downwards
        # print("beforeopt")
        # print(H1)  # the position of a block w.r.t to ROBOT's world frame after rotation
        H1 = opt_pos(H1)
        # print("afteropt")
        # print(H1)  # optimize the pose of the gripper to avoid reaching joint limits
        ###############################
        Htmp = H1.copy()
        Htmp[2][3] += (0.1 + height)
        # the position to hover over the block to avoid toching other blocks when approaching the block
        # this step maybe can be optimized out to save time. But at this step, I still keep this to ensure the process of
        # gripping successful

        if i == 0:  # change the start position of each iteration,  we only need the end configuration, so the start
            # position does not matter much. But we need to select one to avoid reaching joint limits when computing.
            initial_pos = start_position_n #can be changed
        else:
            initial_pos = posdest

        q, success = ik.inverse(Htmp, initial_pos)  # solve IK
          # show information

        ############# Retry process, should be rarely used##################
        if success == False:
            print("Retrying")
            q, success = ik.inverse(Htmp, start_position_n)
            if success == False:
                arm.exec_gripper_cmd(0.2, 50)
                continue
            else:
                print(success)
        else:
            print(success)
        ###########################################################################

        tmppos = q
        arm.safe_move_to_position(tmppos)  # Move to the postion, which is over the block

        #####################################
        q, success = ik.inverse(H1, tmppos)

 ############# Retry process, should be rarely used##################
        if success == False:
            print("Retrying")
            q, success = ik.inverse(H1, start_position_n)
            if success == False:
                arm.exec_gripper_cmd(0.2, 50)
                continue
            else:
                print(success)
        else:
            print(success)
###########################################################################
        prepos = q
        # prepos = pos

        arm.safe_move_to_position(prepos)  # Move to the block!
        arm.exec_gripper_cmd(0.00, 50)  # grip the block
        gstate = arm.get_gripper_state()
        print(gstate)
        if gstate['position'][0] < 0.01:
            print("fail to catch the block")
            arm.exec_gripper_cmd(0.2, 50)
            continue

        arm.safe_move_to_position(tmppos)  # move back to the postion above the block to avoid touching other blocks
        # This step may also be optimized out by firstly moving the most left/right block.

        desheight = int((base + height)*100)
        if team == 'red':
            q = redposdict[desheight]
        else:
            q = blueposdict[desheight]

        arm.safe_move_to_position(q)
        arm.exec_gripper_cmd(0.2, 50)
        q[1] = q[1]-pi/16

        arm.safe_move_to_position(q)  # move over the tower to avoid touching it
        posdest = q
        height += 0.05

#####################################################################################################################################

    if team == 'red':
        startH_D = np.array([[0., 1., -0., -0.15]
                                , [1., 0., 0., 0.68]
                                , [-0., -0., -1., 0.5]
                                , [0., 0., 0.,
                                   1.]])  # # change this and use this in solveik to get the start position: start_position2
        # ydest = 0.68
        ydest = 0.169
        startconfig_D = np.array([1.6743, 0.65995, 0.25624, -0.66686, -0.16135, 1.31402, 1.0517])
        # startconfig_N = np.array([ 0.97662,  0.22038,  0.63358, -1.90847, -0.14896,  2.08231,  0.88649])
        startconfig_N = np.array([1.28822, 0.62151, 0.35068, -1.54818, -0.23783, 2.1274, 0.91926])
        start_position_dest = np.array([-0.14589, 0.1306, -0.16275, -1.36351, 0.02117, 1.49242, 0.47977])

    t_robot = 2
    t_add = 0.7
    T = 100

    arm.exec_gripper_cmd(0.2, 50)

    H_ee_camera = detector.get_H_ee_camera()
    for test_num in range(20):
        if height>0.05*7:
            break
        t_robot = 2
        cubeH_list_D = []
        arm.safe_move_to_position(startconfig_D)
        while (len(cubeH_list_D) == 0):
            for (name, pose) in detector.get_detections():
                print(name, '\n', pose)
                cubeH_list_D.append(pose)

        for i in range(len(cubeH_list_D)):
            HD = startH_D @ H_ee_camera @ cubeH_list_D[i]

            HD1 = HD @ select_rot(HD)  # rotate to make z always point downwards
            if HD1[2][2] > -0.95:  # Used to check whether the block is suitable
                continue
            else:
                print(HD1)
                if HD1[0][3] > -0.1: # change this for more precise prediction, need more test, maybe need a function to describe
                    HD_predicted = predict(HD1, t_robot+0.2, T)
                else:
                    HD_predicted = predict(HD1, t_robot, T)
                print(HD_predicted)
                HD_predicted = opt_pos_D(HD_predicted)
                print(HD_predicted)
                # Htmp = HD_predicted.copy()
                # Htmp[2][3] += (0.1 + height)
                # the position to hover over the block to avoid toching other blocks when approaching the block
                # this step maybe can be optimized out to save time. But at this step, I still keep this to ensure the process of
                # gripping successful

                if i == 0:  # change the start position of each iteration,  we only need the end configuration, so the start
                    # position does not matter much. But we need to select one to avoid reaching joint limits when computing.
                    initial_pos = startconfig_N  # can be changed
                else:
                    initial_pos = startconfig_N

                # q, success = ik.inverse(Htmp, initial_pos)  # solve IK
                # print(success)
                # arm.safe_move_to_position(q)
                q, success = ik.inverse(HD_predicted, initial_pos)
                print(success)
                if success == False:
                    t_robot += t_add
                    continue
                # else:
                #     t_robot = 2
                arm.safe_move_to_position(q)
                arm.exec_gripper_cmd(0.000, 100)
                gstate = arm.get_gripper_state()
                print(gstate)
                if gstate['position'][0] < 0.01 or gstate['position'][1] < 0.01:
                    print("fail to catch the block")
                    arm.exec_gripper_cmd(0.2, 50)
                    break

                q[1] = q[1] - pi / 8 # change to pi / 4 is the tower is tall
                # print(success)
                arm.safe_move_to_position(q)

                desheight = int((base + height) * 100)
                if team == 'red':
                    q = redposdict[desheight]
                else:
                    q = blueposdict[desheight]

                arm.safe_move_to_position(q)
                arm.exec_gripper_cmd(0.2, 50)

                q[1] = q[1] - pi / 16
                arm.safe_move_to_position(q)
                height += 0.05
                break




