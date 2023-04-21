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
# v3.2: 4/21/2023
#       1. The code now works for blue team
#       2. Uodate the prediction function, blue team still needs optimization


import sys
import numpy as np
from copy import deepcopy
from math import pi
from lib.Vec2H import transform, select_rot, opt_pos, predictred, predictblue, opt_pos_D_red, opt_pos_D_blue
import solveIK

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

redposdict = {23: np.array([ 0.26157,  0.2257 ,  0.03175, -2.04018, -0.00925,  2.26575,
        1.08384]), 24: np.array([ 0.2538 ,  0.21183,  0.03987, -2.03145, -0.01071,  2.24309,
        1.08485]), 25: np.array([ 0.24643,  0.19859,  0.04757, -2.02194, -0.01178,  2.22027,
        1.08559]), 26: np.array([ 0.23949,  0.18597,  0.05483, -2.01164, -0.01251,  2.1973 ,
        1.08611]), 27: np.array([ 0.23295,  0.174  ,  0.06166, -2.00056, -0.01296,  2.1742 ,
        1.08643]), 28: np.array([ 0.22682,  0.16267,  0.06805, -1.9887 , -0.01317,  2.15096,
        1.08659]), 29: np.array([ 0.22111,  0.15199,  0.07399, -1.97605, -0.01318,  2.12759,
        1.08662]), 30: np.array([ 0.2158 ,  0.14197,  0.0795 , -1.96261, -0.01305,  2.10411,
        1.08654]), 31: np.array([ 0.2109 ,  0.13263,  0.08457, -1.94838, -0.0128 ,  2.0805 ,
        1.08638]), 32: np.array([ 0.20638,  0.12396,  0.08923, -1.93334, -0.01246,  2.05678,
        1.08615]), 33: np.array([ 0.20227,  0.11598,  0.09347, -1.91749, -0.01207,  2.03294,
        1.08589]), 34: np.array([ 0.19854,  0.10869,  0.0973 , -1.90083, -0.01164,  2.00898,
        1.0856 ]), 35: np.array([ 0.19519,  0.1021 ,  0.10073, -1.88334, -0.0112 ,  1.9849 ,
        1.0853 ]), 36: np.array([ 0.19224,  0.09623,  0.10376, -1.86501, -0.01076,  1.9607 ,
        1.085  ]), 37: np.array([ 0.18966,  0.09108,  0.1064 , -1.84582, -0.01034,  1.93637,
        1.08472]), 38: np.array([ 0.18746,  0.08666,  0.10867, -1.82577, -0.00996,  1.9119 ,
        1.08445]), 39: np.array([ 0.18571,  0.08298,  0.11048, -1.80483, -0.00962,  1.88729,
        1.08421]), 40: np.array([ 0.18428,  0.08006,  0.112  , -1.78298, -0.00933,  1.86253,
        1.08401]), 41: np.array([ 0.18324,  0.07791,  0.11315, -1.76019, -0.00911,  1.83759,
        1.08385]), 42: np.array([ 0.18258,  0.07655,  0.11394, -1.73644, -0.00895,  1.81248,
        1.08373]), 43: np.array([ 0.18232,  0.07598,  0.11436, -1.71169, -0.00887,  1.78717,
        1.08366]), 44: np.array([ 0.18247,  0.07625,  0.11442, -1.68589, -0.00886,  1.76163,
        1.08363]), 45: np.array([ 0.18304,  0.07735,  0.11409, -1.65902, -0.00892,  1.73586,
        1.08366]), 46: np.array([ 0.18403,  0.07933,  0.11339, -1.631  , -0.00905,  1.70981,
        1.08372]), 47: np.array([ 0.1854 ,  0.08221,  0.11237, -1.60178, -0.00927,  1.68347,
        1.08384]), 48: np.array([ 0.1873 ,  0.08603,  0.11088, -1.57128, -0.00954,  1.65678,
        1.08399]), 49: np.array([ 0.18969,  0.09083,  0.10894, -1.53943, -0.00988,  1.62971,
        1.08417]), 50: np.array([ 0.19258,  0.09666,  0.10657, -1.5061 , -0.01027,  1.60221,
        1.08438]), 51: np.array([ 0.19601,  0.10359,  0.1037 , -1.47118, -0.0107 ,  1.57422,
        1.0846 ]), 52: np.array([ 0.2    ,  0.11169,  0.10032, -1.43453, -0.01117,  1.54566,
        1.08482]), 53: np.array([ 0.20459,  0.12106,  0.09637, -1.39595, -0.01164,  1.51645,
        1.08503]), 54: np.array([ 0.20982,  0.13181,  0.09179, -1.35522, -0.01209,  1.48648,
        1.0852 ]), 55: np.array([ 0.21572,  0.1441 ,  0.08653, -1.31205, -0.01249,  1.45562,
        1.08532]), 56: np.array([ 0.22233,  0.15811,  0.08048, -1.26607, -0.0128 ,  1.42369,
        1.08534]), 57: np.array([ 0.22971,  0.17411,  0.07356, -1.21679, -0.01294,  1.39046,
        1.08523]), 58: np.array([ 0.23787,  0.19247,  0.06562, -1.16354, -0.01284,  1.35562,
        1.08495]), 59: np.array([ 0.24686,  0.2137 ,  0.05653, -1.10534, -0.01237,  1.31873,
        1.08442]), 60: np.array([ 0.25669,  0.23861,  0.04606, -1.04074, -0.01136,  1.27912,
        1.08358]), 61: np.array([ 0.26732,  0.26853,  0.03394, -0.96727, -0.00953,  1.23567,
        1.08231]), 62: np.array([ 0.27865,  0.30601,  0.01979, -0.88029, -0.00643,  1.18624,
        1.08051]), 63: np.array([ 0.2903 ,  0.3573 ,  0.00298, -0.76834, -0.00115,  1.12564,
        1.07798]), 64: np.array([ 0.30038,  0.4568 , -0.01765, -0.56847,  0.0091 ,  1.02522,
        1.07467])}


blueposdict = {23: np.array([-0.1161 ,  0.2293 , -0.18308, -2.03992,  0.05386,  2.2648 ,
        0.45644]), 24: np.array([-0.11813,  0.21507, -0.18115, -2.03123,  0.04913,  2.2423 ,
        0.45962]), 25: np.array([-0.12017,  0.20148, -0.17915, -2.02176,  0.04477,  2.21962,
        0.46256]), 26: np.array([-0.12206,  0.18855, -0.17726, -2.0115 ,  0.0408 ,  2.19677,
        0.46524]), 27: np.array([-0.12388,  0.17629, -0.1754 , -2.00045,  0.03716,  2.17376,
        0.4677 ]), 28: np.array([-0.12561,  0.16469, -0.1736 , -1.98861,  0.03386,  2.1506 ,
        0.46994]), 29: np.array([-0.12724,  0.15377, -0.17187, -1.97598,  0.03086,  2.12731,
        0.47197]), 30: np.array([-0.12878,  0.14354, -0.17022, -1.96255,  0.02814,  2.10387,
        0.47382]), 31: np.array([-0.13022,  0.134  , -0.16864, -1.94833,  0.02569,  2.08031,
        0.47549]), 32: np.array([-0.13155,  0.12517, -0.16717, -1.9333 ,  0.02349,  2.05663,
        0.47699]), 33: np.array([-0.13279,  0.11704, -0.1658 , -1.91746,  0.02153,  2.03281,
        0.47833]), 34: np.array([-0.13392,  0.10962, -0.16453, -1.9008 ,  0.01979,  2.00888,
        0.47953]), 35: np.array([-0.13495,  0.10292, -0.16337, -1.88332,  0.01825,  1.98482,
        0.48058]), 36: np.array([-0.13588,  0.09696, -0.16233, -1.86499,  0.01692,  1.96063,
        0.4815 ]), 37: np.array([-0.13666,  0.09173, -0.16148, -1.84581,  0.01577,  1.93631,
        0.4823 ]), 38: np.array([-0.1374 ,  0.08725, -0.16067, -1.82576,  0.01479,  1.91185,
        0.48298]), 39: np.array([-0.13807,  0.08352, -0.15998, -1.80482,  0.01398,  1.88725,
        0.48354]), 40: np.array([-0.13865,  0.08056, -0.15942, -1.78297,  0.01334,  1.86249,
        0.484  ]), 41: np.array([-0.13917,  0.07838, -0.15898, -1.76018,  0.01285,  1.83756,
        0.48435]), 42: np.array([-0.13962,  0.077  , -0.15866, -1.73643,  0.01252,  1.81245,
        0.48459]), 43: np.array([-0.14001,  0.07643, -0.15846, -1.71168,  0.01234,  1.78714,
        0.48473]), 44: np.array([-0.14036,  0.07669, -0.15839, -1.68589,  0.01231,  1.7616 ,
        0.48478]), 45: np.array([-0.14068,  0.0778 , -0.15843, -1.65901,  0.01243,  1.73583,
        0.48472]), 46: np.array([-0.14099,  0.0798 , -0.15858, -1.63099,  0.01271,  1.70978,
        0.48456]), 47: np.array([-0.14129,  0.08271, -0.15886, -1.60177,  0.01315,  1.68343,
        0.4843 ]), 48: np.array([-0.14163,  0.08656, -0.15923, -1.57127,  0.01376,  1.65674,
        0.48394]), 49: np.array([-0.14201,  0.09142, -0.15971, -1.53941,  0.01454,  1.62967,
        0.48348]), 50: np.array([-0.14248,  0.09732, -0.16026, -1.50609,  0.01551,  1.60216,
        0.48291]), 51: np.array([-0.14307,  0.10433, -0.1609 , -1.47117,  0.01668,  1.57415,
        0.48224]), 52: np.array([-0.14383,  0.11253, -0.16157, -1.43451,  0.01807,  1.54558,
        0.48145]), 53: np.array([-0.14482,  0.12201, -0.16228, -1.39592,  0.01969,  1.51635,
        0.48056]), 54: np.array([-0.14617,  0.13289, -0.16289, -1.35518,  0.02157,  1.48636,
        0.47956]), 55: np.array([-0.1478 ,  0.14534, -0.16358, -1.312  ,  0.02375,  1.45546,
        0.47845]), 56: np.array([-0.15   ,  0.15953, -0.16406, -1.26601,  0.02623,  1.42348,
        0.47724]), 57: np.array([-0.15288,  0.17573, -0.16429, -1.21672,  0.02907,  1.3902 ,
        0.47594]), 58: np.array([-0.15665,  0.1943 , -0.16411, -1.16344,  0.0323 ,  1.3553 ,
        0.47458]), 59: np.array([-0.16158,  0.21574, -0.16331, -1.10522,  0.03595,  1.31834,
        0.47321]), 60: np.array([-0.1681 ,  0.24085, -0.16154, -1.04058,  0.04007,  1.27863,
        0.47189]), 61: np.array([-0.17683,  0.27091, -0.15819, -0.96707,  0.04467,  1.23507,
        0.47078]), 62: np.array([-0.18887,  0.30843, -0.15213, -0.88002,  0.04966,  1.18554,
        0.47015]), 63: np.array([-0.20666,  0.35953, -0.1405 , -0.76799,  0.05464,  1.12485,
        0.4707 ]), 64: np.array([-0.24339,  0.45813, -0.10415, -0.56792,  0.05384,  1.02455,
        0.47652])}



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
    '''
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
    '''
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

    if team == 'blue':
        startH_D = np.array([[0, -1, 0, 0.149516011],
                             [-1, 0, 0, -0.680106155],
                             [0, 0, -1, 0.499996060],
                             [0, 0, 0,
                              1]])  # # change this and use this in solveik to get the start position: start_position2
        # ydest = 0.68
        ydest = -0.169
        startconfig_D = np.array([-1.468, 0.65995, 0.25624, -0.66686, -0.16135, 1.31402, 1.0517])
        # startconfig_N = np.array([ 0.97662,  0.22038,  0.63358, -1.90847, -0.14896,  2.08231,  0.88649])
        # startconfig_N = np.array([ -1.468, 0.65995, 0.25624, -0.66686, -0.16135, 1.31402, -1.0517])
        # startconfig_N = np.array([-1.4526, 0.59678, -0.1454, -1.55221, 0.09694, 2.14192, -2.41111])
        startconfig_N = np.array([-1.28822, 0.62151, 0.35068, -1.54818, -0.23783, 2.1274, 0.91926])
        start_position_dest = np.array([-0.14589, 0.1306, -0.16275, -1.36351, 0.02117, 1.49242, 0.47977])

    t_robot = 2.7
    t_add = 0.7
    T = 100

    arm.exec_gripper_cmd(0.2, 50)

    H_ee_camera = detector.get_H_ee_camera()
    for test_num in range(20):
        if height>0.05*8:
            break
        t_robot = 2.5
        mode = 1
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
                if team == 'red':
                    print(HD1)
                    if HD1[0][3] > -0.05:
                        # HD_predicted = predict(HD1, t_robot + 0.4, T)
                        mode = 1
                        HD_predicted = predictred(HD1, t_robot + 1, T)
                        print("situation2")
                        # continue
                    elif HD1[0][3] > -0.1: # change this for more precise prediction, need more test, maybe need a function to describe
                        print("situation1")
                        HD_predicted = predictred(HD1, t_robot+0.4, T)
                    else:
                        HD_predicted = predictred(HD1, t_robot, T)
                    print(HD_predicted)
                    HD_predicted = opt_pos_D_red(HD_predicted, mode)
                    print(HD_predicted)
                else:
                    print(HD1)
                    if HD1[0][3] < 0.05:
                        # HD_predicted = predict(HD1, t_robot + 0.4, T)
                        mode = 1
                        HD_predicted = predictblue(HD1, t_robot + 1, T)
                        print("situation2")
                        # continue
                    elif HD1[0][
                        3] < 0.1:  # change this for more precise prediction, need more test, maybe need a function to describe
                        print("situation1")
                        HD_predicted = predictblue(HD1, t_robot + 0.4, T)
                    else:
                        HD_predicted = predictblue(HD1, t_robot, T)
                    print(HD_predicted)
                    HD_predicted = opt_pos_D_blue(HD_predicted, mode)
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

                q[1] = q[1] - pi / 4 # change to pi / 4 is the tower is tall
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




