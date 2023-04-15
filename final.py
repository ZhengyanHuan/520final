# v1: Preliminary realize stacking static blocks
# v2: 4/15/2023
#       1. Solve the problem that sometimes the gripper cannot grasp the side of a block
#       2. Add the optimization function of make the gripper not rotate too much in the xy-plane
#       3. Raise the height of taking photo of the blocks. This should be changed according to reality.
#       4. Simplify the code of solveIK
#       5. Add some comments
#       6. Possible to optimize: See the comments
import sys
import numpy as np
from copy import deepcopy
from math import pi
from lib.Vec2H import transform, select_rot, opt_pos
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

    
    startH = np.array([[ 1.,     0.,    -0.,     0.562]
    , [ 0.,    -1.,     0.,    -0.169]
    ,[-0.,    -0.,    -1.,     0.6]
    ,[ 0.,    0.,     0.,    1.   ]])

    #start_position2 = np.array([-0.14589, 0.1306, -0.16275, -1.36351, 0.02117, 1.49242, 0.47977])
    start_position2 = np.array([-0.1681,   0.24085, -0.16154, -1.04058,  0.04007,  1.27863, 0.47189])
    # The position to take a photo of the static blocks, should be changed according to real situaltion
    #    target = transform(np.array([0.562,-0.169,0.538]), np.array([0,pi,pi]))
    arm.safe_move_to_position(start_position2)
    arm.exec_gripper_cmd(0.2, 50) #open the gripper

    cubeH_list = []
    # Detect some static blocks...
    for (name, pose) in detector.get_detections():
         print(name,'\n',pose)
         cubeH_list.append(pose)


    ik = solveIK.IK()
    height = 0
    base = 0.23
    for i in range(4):
        H1 = startH@H_ee_camera@cubeH_list[i] # calculate the position of a block w.r.t to ROBOT's world frame
        print("cubeH")
        print(cubeH_list[i]) # the position of a block w.r.t to camera
        print("H1")
        print(H1) # the position of a block w.r.t to ROBOT's world frame
        print(select_rot(H1, 0.999)) # The threshold should be changed according to reality
        H1 = H1@select_rot(H1, 0.999) # rotate to make z always point downwards
        print("beforeopt")
        print(H1)  # the position of a block w.r.t to ROBOT's world frame after rotation
        H1 = opt_pos(H1)
        print("afteropt")
        print(H1)# optimize the pose of the gripper to avoid reaching joint limits
        ###############################
        Htmp = H1.copy()
        Htmp[2][3] += (0.1 + height)
        # the position to hover over the block to avoid toching other blocks when approaching the block
        # this step maybe can be optimized out to save time. But at this step, I still keep this to ensure the process of
        # gripping successful

        if i == 0:  # change the start position of each iteration
            initail_pos = start_position2
        else:
            initail_pos = posdest

        q, success = ik.inverse(Htmp, initail_pos) # solve IK
        print(success) # show information
        tmppos = q
        arm.safe_move_to_position(tmppos) #Move to the postion, which is over the block

        #####################################
        q, success = ik.inverse(H1, tmppos)
        prepos = q
        # prepos = pos

        arm.safe_move_to_position(prepos)# Move to the block!
        arm.exec_gripper_cmd(0.02, 50) # grip the block

        arm.safe_move_to_position(tmppos) # move back to the postion above the block to avoid touching other blocks
        #This step may also be optimized out by firstly moving the most left/right block.


        destpos = np.array([0.562, 0.169, base+height]) # destination of the block (x,y,x)
        # base: the height need for the gripper to avoid touching the block beneath it
        # height: the height of the tower, start with 0, (1 block: 0.05)
        destangle = np.array([0,pi,pi])# destination of the block (angle)
        destH = transform(destpos, destangle)
        q, success = ik.inverse(destH, tmppos)
        # destconfig = np.array([ 0.41668, 0.35412,  0.01618, -1.93268, -0.00744,  2.28674,  1.22214])
        posdest = q
        arm.safe_move_to_position(posdest) # move to destination
        arm.exec_gripper_cmd(0.2, 50) # release the gripper

        destpos = np.array([0.562, 0.169, base + height +0.1])
        destangle = np.array([0, pi, pi])
        destH = transform(destpos, destangle)
        q, success = ik.inverse(destH, posdest)
        posdest = q
        arm.safe_move_to_position(posdest) # move over the tower to avoid touching it

        height+=0.05

    # Move around...

    # END STUDENT CODE

