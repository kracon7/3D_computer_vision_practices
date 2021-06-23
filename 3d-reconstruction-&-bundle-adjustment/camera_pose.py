import numpy as np

from feature import EstimateE_RANSAC
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    U,D,Vh = np.linalg.svd(E)
    
    R = U @ W @ Vh
    t = U[:,2]
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    R1 = R
    C1 = -R.T @ t

    R = U @ W @ Vh
    t = -U[:,2]
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    R2 = R
    C2 = -R.T @ t

    R = U @ W.T @ Vh
    t = U[:, 2]
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    R3 = R
    C3 = -R.T @ t

    R = U @ W.T @ Vh
    t = -U[:, 2]
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    R4 = R
    C4 = -R.T @ t

    R_set = np.stack([R1, R2, R3, R4])
    C_set = np.stack([C1, C2, C3, C4])
    return R_set, C_set

def vec2skew(v):
    skew = np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])
    return skew

def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    n = track1.shape[0]
    X = np.zeros((n, 3))
    for i in range(n):
        u = np.array([track1[i, 0], track1[i, 1], 1])
        v = np.array([track2[i, 0], track2[i, 1], 1])
        A = np.vstack([vec2skew(u) @ P1, vec2skew(v) @ P2])
        U, D, Vh = np.linalg.svd(A)
        p = Vh[-1]
        X[i] = p[:3] / p[3]
    return X



def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    
    homo_X = np.insert(X, 3, 1, axis=1)

    R1, R2 = P1[:, :3], P2[:, :3]
    t1, t2 = P1[:, 3],  P2[:, 3]
    C1, C2 = -R1.T @ t1, -R2.T @ t2
    r3_1, r3_2 = R1[2, :], R2[2, :]

    mask1 = ((X - C1) @ r3_1) > 0
    mask2 = ((X - C2) @ r3_2) > 0

    valid_index = np.logical_and(mask1, mask2).reshape(-1)

    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    F = track1.shape[0]

    # # find pairs for im1 and im2
    mask = np.logical_and(np.sum(track1, axis=1) != -2, np.sum(track2, axis=1) != -2)
    x1, x2 = track1[mask], track2[mask]
    ft_index = np.asarray(np.nonzero(mask)[0])

    E, inlier = EstimateE_RANSAC(x1, x2, 500, 0.003)
    # inlier_index = ft_index[inlier]     # inlier index in all features (.., F, .. )

    R_set, C_set = GetCameraPoseFromE(E)

    num_valid = 0
    validX_set = []
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    for i in range(4):
        P2 = np.hstack([R_set[i], -(R_set[i] @ C_set[i]).reshape((3,1))])
        X_temp = Triangulation(P1, P2, x1, x2)
        cheirality_index = EvaluateCheirality(P1, P2, X_temp)
        validX_set.append(X_temp[cheirality_index])
        # print('Found %d valid points after triangulation'%(np.sum(cheirality_index)))

        if np.sum(cheirality_index) > num_valid:
            num_valid = np.sum(cheirality_index)
            R = R_set[i]
            C = C_set[i]
            X = -1 * np.ones((F,3))
            X[ft_index[cheirality_index]] = X_temp[cheirality_index]

    visualize = 1
    if visualize:
        R0, C0 = np.eye(3), np.zeros(3)

        plt.ion()
        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        R, C = R_set[0], C_set[0]
        draw_camera(ax, R, C)
        draw_camera(ax, R0, C0)
        xx, yy, zz = validX_set[0][:,0], validX_set[0][:,1], validX_set[0][:,2]
        ax.scatter(xx, yy, zz)

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        R, C = R_set[1], C_set[1]
        draw_camera(ax, R, C)
        draw_camera(ax, R0, C0)
        xx, yy, zz = validX_set[1][:,0], validX_set[1][:,1], validX_set[1][:,2]
        ax.scatter(xx, yy, zz)

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        R, C = R_set[2], C_set[2]
        draw_camera(ax, R, C)
        draw_camera(ax, R0, C0)
        xx, yy, zz = validX_set[2][:,0], validX_set[2][:,1], validX_set[2][:,2]
        ax.scatter(xx, yy, zz)

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        R, C = R_set[3], C_set[3]
        draw_camera(ax, R, C)
        draw_camera(ax, R0, C0)
        xx, yy, zz = validX_set[3][:,0], validX_set[3][:,1], validX_set[3][:,2]
        ax.scatter(xx, yy, zz)

    return R, C, X


def draw_camera(ax, R, C, scale=3):
    x_axis, y_axis, z_axis = scale* R[0,:], scale* R[1,:], scale* R[2,:]

    ax.plot3D([C[0], C[0]+x_axis[0]], [C[1], C[1]+x_axis[1]], [C[2], C[2]+x_axis[2]], 'r')
    ax.plot3D([C[0], C[0]+y_axis[0]], [C[1], C[1]+y_axis[1]], [C[2], C[2]+y_axis[2]], 'g')
    ax.plot3D([C[0], C[0]+z_axis[0]], [C[1], C[1]+z_axis[1]], [C[2], C[2]+z_axis[2]], 'b')

    cam_points = scale * np.array([[ 0.5,  0.5, 1.5],
                                   [ 0.5, -0.5, 1.5],
                                   [-0.5, -0.5, 1.5],
                                   [-0.5,  0.5, 1.5]])
    # transform points to camera frame
    pts = cam_points @ R

    ax.plot3D([C[0], C[0]+pts[0,0]], [C[1], C[1]+pts[0,1]], [C[2], C[2]+pts[0,2]], 'k')
    ax.plot3D([C[0], C[0]+pts[1,0]], [C[1], C[1]+pts[1,1]], [C[2], C[2]+pts[1,2]], 'k')
    ax.plot3D([C[0], C[0]+pts[2,0]], [C[1], C[1]+pts[2,1]], [C[2], C[2]+pts[2,2]], 'k')
    ax.plot3D([C[0], C[0]+pts[3,0]], [C[1], C[1]+pts[3,1]], [C[2], C[2]+pts[3,2]], 'k')
    ax.plot3D([C[0]+pts[0,0], C[0]+pts[1,0]], 
              [C[1]+pts[0,1], C[1]+pts[1,1]], 
              [C[2]+pts[0,2], C[2]+pts[1,2]], 'k')
    ax.plot3D([C[0]+pts[1,0], C[0]+pts[2,0]], 
              [C[1]+pts[1,1], C[1]+pts[2,1]], 
              [C[2]+pts[1,2], C[2]+pts[2,2]], 'k')
    ax.plot3D([C[0]+pts[2,0], C[0]+pts[3,0]], 
              [C[1]+pts[2,1], C[1]+pts[3,1]], 
              [C[2]+pts[2,2], C[2]+pts[3,2]], 'k')
    ax.plot3D([C[0]+pts[3,0], C[0]+pts[0,0]], 
              [C[1]+pts[3,1], C[1]+pts[0,1]], 
              [C[2]+pts[3,2], C[2]+pts[0,2]], 'k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-2, 30)

def draw_all(ax, P, X):
    for pose in P:
        R, t = pose[:, :3], pose[:, 3]
        C = -R.T @ t
        draw_camera(ax, R, C)

    mask = np.logical_and( np.logical_and(X[:,0]!=-1, X[:,1]!=-1), X[:,2]!=-1)
    valid_X = X[mask]
    xx, yy, zz = valid_X[:,0], valid_X[:,1], valid_X[:,2]
    ax.scatter(xx, yy, zz)