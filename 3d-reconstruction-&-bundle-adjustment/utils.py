import numpy as np
  

def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion
    
    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """
    
    r11, r12, r13 = R[0][0], R[0][1], R[0][2]
    r21, r22, r23 = R[1][0], R[1][1], R[1][2]
    r31, r32, r33 = R[2][0], R[2][1], R[2][2]

    # computing four sets of solutions
    qw_1 = np.sqrt(1 + r11 + r22 + r33)
    u1 = 1/2 * np.array([qw_1,
                         (r32-r23)/qw_1,
                         (r13-r31)/qw_1,
                         (r21-r12)/qw_1
                         ])

    qx_2 = np.sqrt(1 + r11 - r22 - r33)
    u2 = 1/2 * np.array([(r32-r23)/qx_2,
                         qx_2,
                         (r12+r21)/qx_2,
                         (r31+r13)/qx_2
                         ])

    qy_3 = np.sqrt(1 - r11 + r22 - r33)
    u3 = 1/2 * np.array([(r13-r31)/qy_3,
                         (r12+r21)/qy_3,
                         qy_3,
                         (r23+r32)/qy_3
                         ])

    qz_4 = np.sqrt(1 - r11 - r22 + r33)
    u4 = 1/2* np.array([(r21-r12)/qz_4,
                        (r31+r13)/qz_4,
                        (r32+r23)/qz_4,
                        qz_4
                        ])

    U = [u1,u2,u3,u4]
    idx = np.array([r11+r22+r33, r11, r22, r33]).argmax()
    q = U[idx]
    assert np.abs(np.linalg.norm(q) - 1) < 1e-4
    return q



def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix
    
    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """
    
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    R = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                  [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                  [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])
    assert np.linalg.norm(np.eye(3) - R.T @ R) < 1e-4
    return R