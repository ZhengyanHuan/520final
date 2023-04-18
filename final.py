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
# v3: 4/18/2023
#       1. Preliminary realize stacking dynamic blocks
import sys
import numpy as np
from copy import deepcopy
from math import pi
from lib.Vec2H import transform, select_rot, opt_pos, predict,opt_pos_D
import solveIK

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

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
    base = 0.23
    # STUDENT CODE HERE

    # get the transform from camera to panda_end_effector

    H_ee_camera = detector.get_H_ee_camera()
    # print(H_ee_camera)

    '''
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

        destpos = np.array([0.562, ydest, base + height])  # destination of the block (x,y,x)
        # base: the height need for the gripper to avoid touching the block beneath it
        # height: the height of the tower, start with 0, (1 block: 0.05)
        destangle = np.array([0, pi, pi])  # destination of the block (angle)
        destH = transform(destpos, destangle)
        q, success = ik.inverse(destH, tmppos)
        ############# Retry process, should be rarely used##################
        if success == False:
            print("Retrying")
            q, success = ik.inverse(destH, start_position_n)
            if success == False:
                arm.exec_gripper_cmd(0.2, 50)
                continue
            else:
                print(success)
        else:
            print(success)
        ###########################################################################
        # destconfig = np.array([ 0.41668, 0.35412,  0.01618, -1.93268, -0.00744,  2.28674,  1.22214])
        posdest = q
        arm.safe_move_to_position(posdest)  # move to destination
        arm.exec_gripper_cmd(0.2, 50)  # release the gripper

        destpos = np.array([0.562, ydest, base + height + 0.1])
        destangle = np.array([0, pi, pi])
        destH = transform(destpos, destangle)
        q, success = ik.inverse(destH, posdest)
        ############# Retry process, should be rarely used##################
        if success == False:
            print("Retrying")
            q, success = ik.inverse(destH, start_position_n)
            if success == False:
                arm.exec_gripper_cmd(0.2, 50)
                continue
            else:
                print(success)
        else:
            print(success)
        ###########################################################################
        posdest = q
        arm.safe_move_to_position(posdest)  # move over the tower to avoid touching it

        height += 0.05

#####################################################################################################################################
    '''
    if team == 'red':
        startH_D = np.array([[0., 1., -0., -0.15]
                              , [1., 0., 0., 0.68]
                              , [-0., -0., -1., 0.5]
                              , [0., 0., 0., 1.]]) #  # change this and use this in solveik to get the start position: start_position2
        # ydest = 0.68
        ydest = 0.169
        startconfig_D = np.array([ 1.6743,   0.65995,  0.25624, -0.66686, -0.16135,  1.31402,  1.0517 ])
        # startconfig_N = np.array([ 0.97662,  0.22038,  0.63358, -1.90847, -0.14896,  2.08231,  0.88649])
        startconfig_N = np.array([ 1.28822,  0.62151,  0.35068, -1.54818, -0.23783,  2.1274,   0.91926])
        start_position_dest = np.array([-0.14589, 0.1306, -0.16275, -1.36351, 0.02117, 1.49242, 0.47977])


    t_robot = 2
    T = 100


    arm.exec_gripper_cmd(0.2, 50)

    H_ee_camera = detector.get_H_ee_camera()
    for test_num in range(5):
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
            if HD1[2][2]>-0.95: # Used to check whether the block is suitable
                continue
            else:
                print(HD1)
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
                    t_robot += 0.4
                    continue
                # else:
                #     t_robot = 2
                arm.safe_move_to_position(q)
                arm.exec_gripper_cmd(0.000, 100)
                gstate = arm.get_gripper_state()
                print(gstate)
                if gstate['position'][0] < 0.01:
                    print("fail to catch the block")
                    arm.exec_gripper_cmd(0.2, 50)
                    continue

                q[1] = q[1]-pi/4
                # print(success)
                arm.safe_move_to_position(q)

                destpos = np.array([0.562, ydest, base + height])  # destination of the block (x,y,x)
                # base: the height need for the gripper to avoid touching the block beneath it
                # height: the height of the tower, start with 0, (1 block: 0.05)
                destangle = np.array([0, pi, pi])  # destination of the block (angle)
                destH = transform(destpos, destangle)
                q, success = ik.inverse(destH, start_position_dest)
                print(success)
                arm.safe_move_to_position(q)
                arm.exec_gripper_cmd(0.2, 100)
                q[1] = q[1]-pi/4
                arm.safe_move_to_position(q)
                height+=0.05
                break




