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

def select_rot(t, thres):
    yp = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]) #x=-1
    yn = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    xp = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    xn = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) #y=1
    if t[2][0]<-thres:
        return yp
    elif t[2][0]>thres:
        return yn
    elif t[2][1]<-thres:
        return xn
    elif t[2][1]>thres:
        return xp
    elif t[2][2]>thres:
        return np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    else:
        return np.eye(4)

if __name__ == '__main__':
    print(transform( np.array([-.2, -.3, .5]), np.array([0,pi,pi])))