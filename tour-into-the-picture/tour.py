import os
import cv2
import argparse
import numpy as np
import numpy.linalg as la
from pylsd import lsd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.ion()

def FindVP(lines, K, ransac_thr, ransac_iter):
    """
    Find the vanishing point
    
    Parameters
    ----------
    lines : ndarray of shape (N_l, 4)
        Set of line segments where each row contains the coordinates of two 
        points (x1, y1, x2, y2)
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    ransac_thr : float
        Error threshold for RANSAC
    ransac_iter : int
        Number of RANSAC iterations

    Returns
    -------
    vp : ndarray of shape (2,)
        The vanishing point
    inlier : ndarray of shape (N_i,)
        The index set of line segment inliers
    """
    # compute normalized coordinates: lines_norm
    lines_norm = lines[:,:4].copy()
    N = lines.shape[0]
    # project lines using intrinsic matrix K
    lines_norm[:,:2] = (np.concatenate([lines_norm[:,:2], np.ones((N,1))], axis=1) @
                   la.inv(K).T )[:, :2]
    lines_norm[:,2:4] = (np.concatenate([lines_norm[:,2:4], np.ones((N,1))], axis=1)@ 
                    la.inv(K).T )[:, :2]
    
    vp = np.array([0., 0.])
    inlier = None
    max_num_inlier = 0
    for n_iter in range(ransac_iter):
        '''
        sample two lines to compute vanishing point (vz_x, vz_y)
        compute using: [[a1, b1],
                        [a2, b2]] @ [[vz_x], [vz_y]] = [[c1], [c2]]
        '''
        idx = np.random.choice(np.arange(N), 2)
        l1, l2 = lines_norm[idx[0]], lines_norm[idx[1]]
        x1_1, y1_1, x1_2, y1_2 = l1
        x2_1, y2_1, x2_2, y2_2 = l2
        a1, b1, c1 = y1_2 -y1_1, x1_1 - x1_2, x1_1*y1_2 - x1_2*y1_1
        a2, b2, c2 = y2_2 -y2_1, x2_1 - x2_2, x2_1*y2_2 - x2_2*y2_1
        # normalized coordinates of vanishing point, not pixel coordinates
        vz = la.pinv(np.array([[a1, b1],[a2, b2]])) \
                        @ np.array([[c1],[c2]])
        
        # compute distance between vanishing hypothesis and all line segemnts
        X1, Y1, X2, Y2 = lines_norm[:,0], lines_norm[:,1],lines_norm[:,2],lines_norm[:,3] 
        A, B, C = Y2 - Y1, X1 - X2, X1*Y2 - X2*Y1
        dist = np.abs(A*vz[0] + B*vz[1] - C) / np.sqrt(A**2 + B**2)
        
        # get inliers
        mask = dist < ransac_thr
        n_inlier = np.sum(mask)
        if n_inlier > max_num_inlier:
            inlier = np.array(np.nonzero(mask))
            max_num_inlier = n_inlier
            # compute pixel coordinates of vanishing point
            pz = K @ np.insert(vz, 2, 1, axis=0)
            vp = pz[:2]
    
    return vp.reshape(-1), inlier.reshape(-1)


def ClusterLines(lines):
    """
    Cluster lines into two sets

    Parameters
    ----------
    lines : ndarray of shape (N_l - N_i, 4)
        Set of line segments excluding the inliers from the ﬁrst vanishing 
        point detection

    Returns
    -------
    lines_x : ndarray of shape (N_x, 4)
        The set of line segments for horizontal direction
    lines_y : ndarray of shape (N_y, 4)
        The set of line segments for vertical direction
    """
    X1, Y1, X2, Y2 = lines[:,0], lines[:,1],lines[:,2],lines[:,3] 
    A, B, C = Y2 - Y1, X1 - X2, X1*Y2 - X2*Y1
    slope = np.mod(np.degrees(np.arctan2(np.abs(A), np.abs(B))), 180).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(slope)
    angle_mask = np.logical_or(slope<25, slope>75).reshape(-1)

    vx_mask = np.logical_and(kmeans.labels_==0, angle_mask)
    lines_x = lines[vx_mask]

    vy_mask = np.logical_and(kmeans.labels_==1, angle_mask)
    lines_y = lines[vy_mask]
    return lines_x, lines_y


def CalibrateCamera(vp_x, vp_y, vp_z):
    """
    Calibrate intrinsic parameters

    Parameters
    ----------
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction

    Returns
    -------
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    """

    u1, v1 = vp_z
    u2, v2 = vp_x
    u3, v3 = vp_y
    A = np.array([[u1*u2 + v1*v2, u1+u2, v1+v2, 1],
                  [u3*u2 + v3*v2, u3+u2, v3+v2, 1],
                  [u1*u3 + v1*v3, u1+u3, v1+v3, 1]])
    A = np.vstack(A)
    u, s, v = la.svd(A)
    b = v[-1,:]

    px = - b[1]/b[0]
    py = - b[2]/b[0]
    f = np.sqrt(b[3]/b[0] - (px**2 + py**2))

    K = np.array([[f,  0., px],
                  [0.,  f, py],
                  [0., 0., 1.]])
    return K


def GetRectificationH(K, vp_x, vp_y, vp_z):
    """
    Find a homography for rectification
    
    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction

    Returns
    -------
    H_rect : ndarray of shape (3, 3)
        The rectiﬁcation homography induced by pure rotation
    """
    rx = la.inv(K) @ np.array([vp_x[0], vp_x[1], 1])
    rx = rx / la.norm(rx)

    ry = la.inv(K) @ np.array([vp_y[0], vp_y[1], 1])
    ry = ry / la.norm(ry)

    rz = np.cross(rx, ry)
    ry = np.cross(rz, rx)

    R = np.vstack([rx, ry, rz]).T

    H = K @ R @ la.inv(K)
    return H


def ImageWarping(im, H):
    """
    Warp image by the homography

    Parameters
    ----------
    im : ndarray of shape (h, w, 3)
        Input image
    H : ndarray of shape (3, 3)
        Homography

    Returns
    -------
    im_warped : ndarray of shape (h, w, 3)
        The warped image
    """
    # pixels of image grid
    im_h, im_w = im.shape[:2]
    x, y = np.arange(im_w), np.arange(im_h)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy], axis=2).reshape(-1,2)

    # project pixels
    projected = np.insert(points, 2, 1, axis=1) @ H.T
    projected = (projected / projected[:,-1:])[:,:2]
    mask = np.logical_and(
            np.logical_and(projected[:, 0]<im_w, projected[:, 0]>0),
            np.logical_and(projected[:, 1]<im_h, projected[:, 1]>0))
    
    canvas = np.zeros((im_h, im_w, 3))
    canvas = canvas.reshape(-1, 3)
    # valid inbound pixels coordinates of original image grid
    pixels = projected[mask].astype('int')
    canvas[mask>0] = im[pixels[:,1], pixels[:,0], :]
    canvas = canvas.reshape(im_h, im_w, 3).astype('uint8')
    return canvas

def ConstructBox(K, vp_x, vp_y, vp_z, W, a, d_near, d_far):
    """
    Construct a 3D box to approximate the scene geometry
    
    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction
    W : float
        Width of the box
    a : float
        Aspect ratio
    d_near : float
        Depth of the front plane
    d_far : float
        Depth of the back plane

    Returns
    -------
    U11, U12, U21, U22, V11, V12, V21, V22 : ndarray of shape (3,)
        The 8 corners of the box
    """
    # z direction 
    z_hat = la.inv(K) @ np.array([vp_z[0], vp_z[1], 1])
    z_hat = z_hat / z_hat[2]
    x_hat = la.inv(K) @ np.array([vp_x[0], vp_x[1], 1])
    x_hat = x_hat / la.norm(x_hat)
    y_hat = la.inv(K) @ np.array([vp_y[0], vp_y[1], 1])
    y_hat = y_hat / la.norm(y_hat)

    near_depth = d_near
    far_depth = d_far
    aspect_ratio = a
    Wd= W
    Ht = Wd / aspect_ratio

    # find 4 cornors of near plane
    vz = near_depth * la.inv(K) @ np.array([vp_z[0], vp_z[1], 1])
    
    U11 = vz + Wd * x_hat + Ht * y_hat
    U12 = vz + Wd * x_hat - Ht * y_hat
    U13 = vz - Wd * x_hat + Ht * y_hat
    U14 = vz - Wd * x_hat - Ht * y_hat

    # find 4 cornors of far plane
    vz = far_depth * la.inv(K) @ np.array([vp_z[0], vp_z[1], 1])

    U21 = vz + Wd * x_hat + Ht * y_hat
    U22 = vz + Wd * x_hat - Ht * y_hat
    U23 = vz - Wd * x_hat + Ht * y_hat
    U24 = vz - Wd * x_hat - Ht * y_hat
    
    return U11, U12, U13, U14, U21, U22, U23, U24       



def InterpolateCameraPose(R1, C1, R2, C2, w):
    """
    Interpolate the camera pose
    
    Parameters
    ----------
    R1 : ndarray of shape (3, 3)
        Camera rotation matrix of camera 1
    C1 : ndarray of shape (3,)
        Camera optical center of camera 1
    R2 : ndarray of shape (3, 3)
        Camera rotation matrix of camera 2
    C2 : ndarray of shape (3,)
        Camera optical center of camera 2
    w : float
        Weight between two poses

    Returns
    -------
    Ri : ndarray of shape (3, 3)
        The interpolated camera rotation matrix
    Ci : ndarray of shape (3,)
        The interpolated camera optical center
    """
    q1 = Rotation2Quaternion(R1)
    q2 = Rotation2Quaternion(R2)
    # get sphere angle between two quaternions
    omega = np.arccos(q1[0] * q2[0] - q1[1:] @ q2[1:])
    if omega > 1e-3:
        # get interpolated quaternion
        p = (q1 * np.sin((1-w)*omega) + q2 * np.sin(w*omega)) / np.sin(omega)
    else:
        p = q1
    
    # normalize p
    p = p / la.norm(p)

    Ri = Quaternion2Rotation(p)
    
    Ci = (1-w) * C1 + w * C2
    
    return Ri, Ci


def Rotation2Quaternion(R):
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
    assert (np.linalg.norm(q) - 1) < 1e-4
    return q

def Quaternion2Rotation(q):
    
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    R = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                  [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                  [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])
    assert np.linalg.norm(np.eye(3) - R.T @ R) < 1e-4
    return R

def GetPlaneHomography(p11, p12, p21, K, R, C, vx, vy):
    """
    Interpolate the camera pose
    
    Parameters
    ----------
    p11 : ndarray of shape (3,)
        Top-left corner
    p12 : ndarray of shape (3,)
        Top-right corner
    p21 : ndarray of shape (3,)
        Bottom-left corner
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    R : ndarray of shape (3, 3)
        Camera rotation matrix
    C : ndarray of shape (3,)
        Camera optical center
    vx : ndarray of shape (h, w)
        All x coordinates in the image
    vy : ndarray of shape (h, w)
        All y coordinates in the image

    Returns
    -------
    H : ndarray of shape (3, 3)
        The homography that maps the rectiﬁed image to the canvas
    visibility_mask : ndarray of shape (h, w)
        The binary mask indicating membership to the plane constructed by p11, 
        p12, and p21
    """
    
    # compute upper bound of mu for each directions
    mu1_max = la.norm(p12 - p11)
    mu2_max = la.norm(p21 - p11)
    
    # get three axis vector for the plane
    x_hat = (p12-p11) / mu1_max
    y_hat = (p21-p11) / mu2_max
    z_hat = np.cross(x_hat, y_hat)
    
    # compute two homography matrix
    H_hat = K @ np.vstack([x_hat, y_hat, p11]).T
    H_tilda = K @ np.vstack([R.T@x_hat, R.T@y_hat, R.T@p11 - C]).T
    
    im_h, im_w = vx.shape[0], vx.shape[1]
    visibility_mask = np.zeros(im_h*im_w)
    
    pixels = np.stack([vx.reshape(-1), vy.reshape(-1), np.ones(im_h*im_w)], axis=1)
    rays = pixels @ la.pinv(H_tilda).T
    
    mu1, mu2 = rays[:,0] / rays[:,2], rays[:,1] / rays[:,2]
    pix_mask = np.logical_and(rays[:,2] > 0, np.logical_and(
                    np.logical_and(mu1 > 0, mu1 < mu1_max),
                    np.logical_and(mu2 > 0, mu2 < mu2_max)))
    
    
    visibility_mask = pix_mask.reshape(im_h, im_w)
    
    return H_hat @ la.inv(H_tilda), visibility_mask
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    os.system('mkdir -p %s'%(args.output_dir))

    # Load the input image and detect the line segments
    im = cv2.imread(args.img_path)
    im_h = im.shape[0]
    im_w = im.shape[1]
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lines = lsd(im_gray)

    # Approximate K
    f = 300
    K_apprx = np.asarray([
        [f, 0, im_w/2],
        [0, f, im_h/2],
        [0, 0, 1]
    ])

    #####################################################################
    # Compute the major z-directional vanishing point and its line segments using approximate K
    vp_z, inlier = FindVP(lines, K_apprx, 0.04, 3000)

    n_inlier = inlier.shape[0]
    
    #####################################################################
    # Cluster the rest of line segments into two major directions and compute the x- and y-directional vanishing points using approximate K
    N = lines.shape[0]
    inlier_set = set(inlier.tolist())
    whole_set = set([i for i in range(N)])
    outlier_set = whole_set.difference(inlier_set)
    outlier = np.array(list(outlier_set))
    outlier_lines = lines[outlier, :4]
    lines_x, lines_y = ClusterLines(outlier_lines)

    vp_x, inl_x = FindVP(lines_x, K_apprx, 0.45, 30000)
    vp_y, inl_y = FindVP(lines_y, K_apprx, 0.1, 20000)

    #####################################################################
    # Calibrate K 
    K = CalibrateCamera(vp_x, vp_y, vp_z)

    #####################################################################
    # Compute the rectiﬁcation homography
    H = GetRectificationH(K, vp_x, vp_y, vp_z)

    #####################################################################
    # Rectify the input image and vanishing points
    rect_im = ImageWarping(im, H)

    #####################################################################
    # Construct 3D representation of the scene using a box model
    W = 0.4
    aspect_ratio = 2.5
    near_depth = 0.4
    far_depth = 2
    
    corners = ConstructBox(K, vp_x, vp_y, vp_z, W, aspect_ratio, near_depth, far_depth)

    R0 = np.eye(3)
    C0 = np.zeros(3)

    R1 = np.asarray([
            [np.cos(np.pi/12), 0, np.sin(np.pi/12)],
            [0, 1, 0],
            [-np.sin(np.pi/12), 0, np.cos(np.pi/12)]
        ])
    C1 = np.asarray([0, -0.05, 0.3])

    R2 = np.asarray([
            [np.cos(np.pi/6), 0, -np.sin(np.pi/6)],
            [0, 1, 0],
            [np.sin(np.pi/6), 0, np.cos(np.pi/6)]
        ])
    C2 = np.asarray([-0.05, 0, 0.6])

    R3 = np.asarray([
            [np.cos(np.pi/12), 0, -np.sin(-np.pi/12)],
            [0, 1, 0],
            [np.sin(np.pi/12), 0, np.cos(np.pi/12)]
        ])
    C3 = np.asarray([-0.03, 0, 0.6])

    R4 = np.asarray([
            [np.cos(np.pi/12), 0, -np.sin(np.pi/12)],
            [0, 1, 0],
            [np.sin(np.pi/12), 0, np.cos(np.pi/12)]
        ])
    C4 = np.asarray([0, -0.02, 0.9])

    # The sequence of camera poses
    R_list = []
    C_list = []
    for w in np.linspace(0, 1, 10):
        Ri, Ci = InterpolateCameraPose(R0, C0, R1, C1, w)
        R_list.append(Ri)
        C_list.append(Ci)

    for w in np.linspace(0, 1, 10):
        Ri, Ci = InterpolateCameraPose(R1, C1, R2, C2, w)
        R_list.append(Ri)
        C_list.append(Ci)

    for w in np.linspace(0, 1, 10):
        Ri, Ci = InterpolateCameraPose(R2, C2, R3, C3, w)
        R_list.append(Ri)
        C_list.append(Ci)

    for w in np.linspace(0, 1, 10):
        Ri, Ci = InterpolateCameraPose(R3, C3, R4, C4, w)
        R_list.append(Ri)
        C_list.append(Ci)


    #####################################################################
    # Render images from the interpolated virtual camera poses 
    plane_point_list = [[corners[7], corners[5], corners[6]],
                        [corners[3], corners[1], corners[7]],
                        [corners[6], corners[4], corners[2]],
                        [corners[5], corners[1], corners[4]],
                        [corners[3], corners[7], corners[2]]]

    x, y = np.arange(im_w), np.arange(im_h)
    vx, vy = np.meshgrid(x, y)

    canvas_list = []

    for R, C in zip(R_list, C_list):
        canvas = np.zeros((im_h, im_w, 3)).astype('uint8')
        
        # map each plane    
        for i, points in enumerate(plane_point_list):
            p11, p12, p21 = points
            H, mask = GetPlaneHomography(p11, p12, p21, K, R, C, vx, vy)

            warpped = ImageWarping(im, H)
            canvas += warpped * np.stack([mask]*3, axis=2).astype('uint8')
            # ax[i+1].imshow(canvas)

        canvas_list.append(canvas)

    # visualization and generate gif
    import matplotlib.animation as animation
    fig = plt.figure()

    ani_ims = []
    for i, img in enumerate(canvas_list):
        cv2.imwrite(os.path.join(args.output_dir, '%d.png'%(i)), img)
        ani_im = plt.imshow(img, animated=True)
        ani_ims.append([ani_im])
    ani = animation.ArtistAnimation(fig, ani_ims, interval=100, blit=True, repeat_delay=1000)
    video_path = os.path.join(args.output_dir, 'tour.gif')
    ani.save(video_path)