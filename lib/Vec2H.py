import numpy as np
from math import sin,cos,pi


def trans(d):
    """
    Compute pure translation homogenous transformation
    """
    return np.array([
        [ 1, 0, 0, d[0] ],
        [ 0, 1, 0, d[1] ],
        [ 0, 0, 1, d[2] ],
        [ 0, 0, 0, 1    ],
    ])

def roll(a):
    """
    Compute homogenous transformation for rotation around x axis by angle a
    """
    return np.array([
        [ 1,     0,       0,  0 ],
        [ 0, cos(a), -sin(a), 0 ],
        [ 0, sin(a),  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def pitch(a):
    """
    Compute homogenous transformation for rotation around y axis by angle a
    """
    return np.array([
        [ cos(a), 0, -sin(a), 0 ],
        [      0, 1,       0, 0 ],
        [ sin(a), 0,  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def yaw(a):
    """
    Compute homogenous transformation for rotation around z axis by angle a
    """
    return np.array([
        [ cos(a), -sin(a), 0, 0 ],
        [ sin(a),  cos(a), 0, 0 ],
        [      0,       0, 1, 0 ],
        [      0,       0, 0, 1 ],
    ])

def transform(d,rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotation corresponding to roll-pitch-yaw euler angles
    """
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])

def select_rot(t):
    zval = max(abs(t[2,0:3]))
    yp = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]) #x=-1
    yn = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    xp = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    xn = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) #y=1
    if t[2][0]==-zval:
        return yp
    elif t[2][0]==zval:
        return yn
    elif t[2][1]==-zval:
        return xn
    elif t[2][1]==zval:
        return xp
    elif t[2][2]==zval:
        return np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    else:
        return np.eye(4)


def opt_pos(H):
    R90 = np.array([[0,-1],[1,0]])
    Rxy = H[0:2,0:2]
    xmax = max(abs(H[0][0]), abs(H[1][0]))
    while (Rxy[0][0] != xmax):
        Rxy = R90@Rxy
    H[0:2,0:2] = Rxy
    return H

# def opt_pos_D(H):
#     R90 = np.array([[0,-1],[1,0]])
#     Rxy = H[0:2,0:2]
#     xmax = max(abs(H[0][0]), abs(H[1][0]))
#     if H[0][3] <= -0.15:
#         while (Rxy[1][0] != xmax):
#             Rxy = R90 @ Rxy
#     else:
#         while Rxy[0][0] >= 0 or Rxy[1][0]<=0:
#             Rxy = R90 @ Rxy
#     H[0:2,0:2] = Rxy
#     return H

def opt_pos_D(H):
    R90 = np.array([[0,-1],[1,0]])
    Rxy = H[0:2,0:2]
    xmax = max(abs(H[0][0]), abs(H[1][0]))
    while (Rxy[1][0] != xmax):
        Rxy = R90 @ Rxy
    H[0:2,0:2] = Rxy
    return H


def predict(H, t_robot, T): #t_robot is the time needed for the end-effector to reach the table, T is the time need for the table to rotate 2pi
    theta = t_robot/T * 2 * pi
    H[0:2,:] = rot_2D(theta, H[0:2,:])
    return H

def rot_2D(theta, mat): # Reduced dimension for smaller computation complexity
    # xo = 0 # already taken into consideration
    yo = 0.99
    # dx = xo - xo *cos(theta) + yo*sin(theta)
    # dy = yo - yo * cos(theta) - xo *sin(theta)
    s = sin(theta)
    c = cos(theta)
    dx = yo * s
    dy = yo - yo * c

    rotmat = np.array([[c,-s],[s,c]])

    return rotmat@mat+ np.array([[0,0,0,dx],[0,0,0,dy]])



if __name__ == '__main__':
    print(transform( np.array([-.2, -.3, .5]), np.array([0,pi,pi])))