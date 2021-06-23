import numpy as np
from scipy.optimize import least_squares

from pnp import ComputePoseJacobian
from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def FindMissingReconstruction(X, track_i):
    """
    Find the points that will be newly added

    Parameters
    ----------
    X : ndarray of shape (F, 3)
        3D points
    track_i : ndarray of shape (F, 2)
        2D points of the newly registered image

    Returns
    -------
    new_point : ndarray of shape (F,)
        The indicator of new points that are valid for the new image and are 
        not reconstructed yet
    """
    
    X_mask = np.logical_and( np.logical_and(X[:,0]!=0, X[:,1]!=0), X[:,2]!=0)

    track_mask = np.logical_and(track_i[:,0]!= -1, track_i[:,1]!= -1)
    mask = np.logical_and(X_mask, track_mask)

    F = X.shape[0]
    new_point = np.zeros(F)
    new_point[mask] = 1

    return new_point

def ComputeTriangulationError(X, P1, P2, b):
    '''
    Compute averaged nonlinear triangulation error E  and vector f for each point in X
    '''
    homo_X = np.insert(X, 3, 1, axis=1)
    x1 = homo_X @ P1.T
    x1 = x1[:, :2] / x1[:,-1:]
    x2 = homo_X @ P2.T
    x2 = x2[:, :2] / x2[:,-1:]

    error1 = np.average(np.linalg.norm(x1 - b[:, :2], axis=1))
    error2 = np.average(np.linalg.norm(x2 - b[:, 2:], axis=1))
    error = (error1 + error2) / 2

    f = np.hstack([x1, x2])      # n x 4
    return error, f


def CompressP(P):
    '''
    compress (3, 4) projection matrix to (7), C + quaternion
    '''
    R = P[:, :3]
    q = Rotation2Quaternion(R)
    t = P[:, 3]
    C = -R.T @ t
    p = np.concatenate([C, q])
    return p


def Triangulation_nl(X, P1, P2, x1, x2):
    """
    Refine the triangulated points

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        3D points
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    x1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    x2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X_new : ndarray of shape (n, 3)
        The set of refined 3D points
    """
    
    # maximum number of iterations
    max_iter = 50
    # threshold for terminating gradient update
    epsilon = 5e-5
    lmbd = 1e-2
    n = x1.shape[0]
    b = np.hstack([x1, x2])  # n x 4
    
    p1, p2 = CompressP(P1), CompressP(P2)
    
    error_prev, f = ComputeTriangulationError(X, P1, P2, b)

    for i_iter in range(max_iter):
        print('Running nonlinear triangulation iteration %d'%(i_iter))
        
        for i in range(n):
            # compute jacobian matrix for each point and stack them
            dfdX = []
            dfdX.append(ComputePointJacobian(X[i], p1))
            dfdX.append(ComputePointJacobian(X[i], p2))
            dfdX = np.vstack(dfdX)     
            assert dfdX.shape[0] == 4 and dfdX.shape[1] == 3

            # compute dX and update position
            #                 | 3x4 |   |4x3|                     | 3x4 |   |  4  |
            dX = np.linalg.inv(dfdX.T @ dfdX + lmbd * np.eye(3)) @ dfdX.T @ (b[i] - f[i])
            X[i] += dX 

        error, f = ComputeTriangulationError(X, P1, P2, b)
        X_new = X

        if error_prev - error < epsilon:
            break
        else:
            error_prev = error
    return X_new


def ComputePointJacobian(X, p):
    """
    Compute the point Jacobian

    Parameters
    ----------
    X : ndarray of shape (3,)
        3D point
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion

    Returns
    -------
    dfdX : ndarray of shape (2, 3)
        The point Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)

    uvw = R @ (X - C)
    duvw_dX = R

    u, v, w = uvw[0], uvw[1], uvw[2]
    du_dX, dv_dX, dw_dX = duvw_dX[0], duvw_dX[1], duvw_dX[2]

    dfdX = np.stack([(w*du_dX - u*dw_dX)/ w**2 , (w*dv_dX - v*dw_dX)/w**2])
    assert dfdX.shape[0] == 2 and dfdX.shape[1] == 3

    return dfdX



def SetupBundleAdjustment(P, X, track):
    """
    Setup bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    z : ndarray of shape (7K+3J,)
        The optimization variable that is made of all camera poses and 3D points
    b : ndarray of shape (2M,)
        The 2D points in track, where M is the number of 2D visible points
    S : ndarray of shape (2M, 7K+3J)
        The sparse indicator matrix that indicates the locations of Jacobian computation
    camera_index : ndarray of shape (M,)
        The index of camera for each measurement
    point_index : ndarray of shape (M,)
        The index of 3D point for each measurement
    """
    
    K, J = P.shape[0], X.shape[0]

    z = []
    for i in range(K):
        z.append(CompressP(P[i]))
    for i in range(J):
        z.append(X[i])
    z = np.hstack(z)

    track_mask = np.logical_and(track[:,:,0]!= -1, track[:,:,1]!= -1)
    point_index, camera_index = np.nonzero(track_mask.transpose(1,0))

    b = track.transpose(1,0,2)[track_mask.transpose(1,0)].reshape(-1)
    M = int(b.shape[0] / 2)

    S = np.zeros((2*M, 7*K+3*J))
    for m in range(M):
        cam_id, pt_id = camera_index[m], point_index[m]
        # do not update the first two camera poses through Jacobian
        if cam_id != 0 and cam_id != 1:
            S[2*m : 2*(m+1), 7*cam_id : 7*(cam_id+1)] = 1
        S[2*m : 2*(m+1), 7*K + 3*pt_id : 7*K + 3*(pt_id+1)] = 1

    return z, b, S, camera_index, point_index
    


def MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index):
    """
    Evaluate the reprojection error

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    b : ndarray of shape (2M,)
        2D measured points
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points
    camera_index : ndarray of shape (M,)
        Index of camera for each measurement
    point_index : ndarray of shape (M,)
        Index of 3D point for each measurement

    Returns
    -------
    err : ndarray of shape (2M,)
        The reprojection error
    """
    
    P, X = UpdatePosePoint(z, n_cameras, n_points)
    
    rays = []
    for c_idx, p_idx in zip(camera_index, point_index):
        ray = P[c_idx] @ np.insert(X[p_idx], 3, 1)
        rays.append(ray[:2] / ray[2])
    rays = np.vstack(rays).reshape(-1)

    err = np.abs(rays - b)

    return err



def UpdatePosePoint(z, n_cameras, n_points):
    """
    Update the poses and 3D points

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points

    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    
    P_new = np.zeros((n_cameras, 3, 4))
    for i in range(n_cameras):
        p = z[7*i : 7*(i+1)]
        C = p[:3]
        R = Quaternion2Rotation( p[3:] / np.linalg.norm(p[3:]) )
        t = - R @ C
        P_new[i] = np.hstack([R, t.reshape((3,1))])

    X_new = z[7*n_cameras:].reshape((n_points, 3))

    return P_new, X_new

def F(z, b, n_cameras, n_points, camera_index, point_index):
    err = MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index)
    return np.average(err)


def RunBundleAdjustment(P, X, track):
    """
    Run bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """

    n_cameras, n_points = P.shape[0], X.shape[0]
    print('Running bundle adjustment for %d cameras and %d points'%(n_cameras, n_points))
    z0, b, S, camera_index, point_index = SetupBundleAdjustment(P, X, track)
    res = least_squares(MeasureReprojection, z0, args=(b, n_cameras, n_points, camera_index, point_index), 
                    jac_sparsity=S)

    P_new, X_new = UpdatePosePoint(res.x, n_cameras, n_points)
    return P_new, X_new