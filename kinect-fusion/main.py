import os
import argparse
import numpy as np
import numpy.linalg as la
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from utils import *
from rays import *
from tsdf import *
import cv2

plt.ion()
np.set_printoptions(suppress=True)

def ProcessDepthImage(file_name, depth_factor):
    """
    Process Depth Image

    Parameters
    ----------
    filename : string
        input depth file
    depth_factor : float
        normalized depth value

    Returns
    -------
    depth_img : ndarray of shape (480, 640)
        filtered depth image
    """
    depth_img = Image.open(file_name).convert('F')
    depth_img = np.array(depth_img) / depth_factor
    scale = np.max(depth_img)
    d_ = depth_img / scale
    d_ = cv2.bilateralFilter(d_, 5, 3, 0.01)
    depth_img = d_ * scale
    return depth_img


def Get3D(depth, K):
    """
        Inverse Projection - create point cloud from depth image

        Parameters
        ----------
        depth : ndarray of shape (H, W)
            filtered depth image
        K : ndarray of shape (3, 3)
            Intrinsic parameters
        Returns
        -------
        point : ndarray of shape (3, H, W)
            Point cloud from depth image
        normal : ndarray of shape (3, H, W)
            Surface normal
    """

    im_h, im_w = depth.shape[:2]
    x, y = np.arange(im_w), np.arange(im_h)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy], axis=2).reshape(-1,2)

    # project pixels
    rays = np.insert(points, 2, 1, axis=1) @ la.inv(K).T
    point = rays.reshape(im_h, im_w, 3) * np.expand_dims(depth, axis=-1)  # im_h x im_w x 3
    point = point.transpose(2, 0, 1)

    padded = np.pad(point, ((0,0),(0,1),(0,1)), 'edge')
    mask = (padded[2, 1:, :-1]==0) | (padded[2, :-1, 1:]==0) | (padded[2, :-1, :-1]==0)
    
    # compute surface normal
    vx = padded[:, 1:, :-1] - padded[:, :-1, :-1]
    vy = padded[:, :-1, 1:] - padded[:, :-1, :-1]
    n = np.cross(vx, vy, axisa=0, axisb=0, axisc=0)
    normal = n / la.norm(n, axis=0, keepdims=True)

    point[:, mask] = 0
    normal[:, mask] = 0
    return point, normal


def CreateTSDF(depth, T, voxel_param, K):
    """
        CreateTSDF : VoxelParams class' member function
            Compute distance of each voxel w.r.t a camera

        Parameters
        ----------
        depth : ndarray of shape (H, W)
            Filtered depth image
        T : ndarray of shape (4, 4)
            Transformation that brings camera to world coordinate
        voxel_param : an instance of voxel parameter VoxelParams
        K : ndarray of shape (3, 3)
                Intrinsic parameters
        Returns
        -------
        tsdf : TSDF
            An instance of TSDF with value computed as projective TSDF
    """

    num_x = voxel_param.num_x
    # initialize sdf
    sdf = -100*np.ones((num_x*num_x*num_x))

    # compute voxel to cam center distance
    pt_world = np.vstack([voxel_param.voxel_x.reshape(-1),
                          voxel_param.voxel_y.reshape(-1),
                          voxel_param.voxel_z.reshape(-1)])   # (num_x * num_x * num_x, 3)
    pt_cam = inverse_trans(T) @ np.insert(pt_world, 3, 1, axis=0)    # (num_x * num_x * num_x, 4)
    dist_v = la.norm(pt_cam[:3], axis=0)

    # compute depth to cam center distance
    im_h, im_w = depth.shape[:2]
    x, y = np.arange(im_w), np.arange(im_h)
    xx, yy = np.meshgrid(x, y)
    rays = np.stack([xx, yy, np.ones((im_h, im_w))], axis=2).reshape(-1,3) @ la.inv(K).T
    points = rays.reshape(im_h, im_w, 3) * np.expand_dims(depth, axis=-1)  # im_h x im_w x 3
    dist_p = la.norm(points, axis=-1)
    
    # compute valid mask    
    ray_vox = pt_cam[:2] / pt_cam[2]        
    pixel_vox = K @ np.insert(ray_vox, 2, 1, axis=0)
    pixel_vox = pixel_vox[:2].astype('int')

    mask1 = (pixel_vox[0]>=0) & (pixel_vox[1]>=0) & (pixel_vox[0]<im_w) & (pixel_vox[1]<im_h)
    # compute voxel with corresponding pixels within camera field of view
    fov_voxel_idx = np.nonzero(mask1)[0]
    # compute mask for fov pixels with nonzero depth
    mask2 = depth[pixel_vox[1][mask1], pixel_vox[0][mask1]] != 0
    valid_idx = fov_voxel_idx[mask2]

    sdf[valid_idx] = dist_p[pixel_vox[1][valid_idx], pixel_vox[0][valid_idx]] - dist_v[valid_idx]

    sdf = sdf.reshape(num_x, num_x, num_x)
    mask = np.zeros((num_x*num_x*num_x))
    mask[valid_idx] = 1
    mask = mask.reshape(num_x, num_x, num_x).astype('bool')
    tsdf = TSDF(voxel_param, sdf, mask)

    return tsdf


def ComputeTSDFNormal(point, tsdf, voxel_param):
    """
        ComputeTSDFNormal : Compute surface normal from tsdf


        Parameters
        ----------
        point : ndarray of shape (3, H, W)
            Point cloud predicted by casting rays to tsdf
        voxel_param : an instance of voxel parameter VoxelParams
        tsdf : an instance of TSDF

        Returns
        -------
        normal : ndarray of shape (3, H, W)
            Surface normal at each 3D point indicated by 'point' variable

        Note
        -------
        You can use scipy.ndimage.map_coordinates to interpolate ndarray
    """
    # pt = point.transpose(1,2,0)     # (H, W, 3)
    # pt_mask = pt[:,:,2] != 0
    _, H, W = point.shape
    coord = (point - voxel_param.voxel_origin.reshape(3,1,1)) / voxel_param.vox_size
    
    vp = ndimage.map_coordinates(voxel_param.trunc_thr*tsdf.value, coord.reshape(3,-1), order=1)

    coord_x, coord_y, coord_z = coord.copy(), coord.copy(), coord.copy()
    coord_x[0,:,:] += 1
    coord_y[1,:,:] += 1
    coord_z[2,:,:] += 1

    vpx = ndimage.map_coordinates(voxel_param.trunc_thr*tsdf.value, coord_x.reshape(3,-1), order=1)
    vpy = ndimage.map_coordinates(voxel_param.trunc_thr*tsdf.value, coord_y.reshape(3,-1), order=1)
    vpz = ndimage.map_coordinates(voxel_param.trunc_thr*tsdf.value, coord_z.reshape(3,-1), order=1)

    npx, npy, npz = vpx - vp, vpy - vp, vpz - vp
    normal = np.vstack([npx, npy, npz])
    normal = normal / (la.norm(normal, axis=0) + 1e-5)
    normal = normal.reshape(3, H, W)

    return normal


def FindCorrespondence(T, point_pred, normal_pred, point, normal, valid_rays, K, e_p, e_n):
    """
    Find Correspondence between current tsdf and input image's depth/normal

    Parameters
    ----------
    T : ndarray of shape (4, 4)
        Transformation of camera to world coordinate
    point_pred : ndarray of shape (3, H, W)
        Point cloud from ray casting the tsdf
    normal_pred : ndarray of shape (3, H, W)
        Surface normal from tsdf
    point : ndarray of shape (3, H, W)
        Point cloud extracted from depth image
    normal : ndarray of shape (3, H, W)
        Surface normal extracted from depth image
    valid_rays : ndarray of shape (H, W)
        Valid ray casting pixels
    K : ndarray of shape (3, 3)
        Intrinsic parameters
    e_p : float
        Threshold on distance error
    e_n : float
        Threshold on cosine angular error
    Returns
    -------
    Correspondence point of 4 variables
    p_pred, n_pred, p, n : ndarray of shape (3, m)
        Inlier point_pred, normal_pred, point, normal

    """
    _, H, W = point_pred.shape
    w_R_c, w_C = T[:3, :3], T[:3, 3]

    p_c = w_R_c.T @ (point_pred.reshape(3,-1) - w_C.reshape(3,1))
    n_c = w_R_c.T @ normal_pred.reshape(3,-1)

    point_mask = ~((p_c[0]==0)&(p_c[1]==0)&(p_c[2]==0)) | (p_c[2]<0)

    pixels_pred = (K @ (p_c / p_c[2]))[:2].astype('int')
    pixel_mask = (pixels_pred[0]>=0) & (pixels_pred[1]>=0) & \
                 (pixels_pred[0]<=W) & (pixels_pred[1]<=H)

    point_correspondence = la.norm(p_c-point.reshape(3,-1), axis=0) < e_p
    normal_correspondence = np.sum(n_c * normal.reshape(3,-1), axis=0) > np.cos(e_n)

    mask = point_mask & pixel_mask & point_correspondence #& normal_correspondence
    p_pred, n_pred = point_pred.reshape(3,-1)[:, mask], normal_pred.reshape(3,-1)[:, mask]

    # transform target points from cam frame to world frame
    point_target_w = w_R_c @ point.reshape(3,-1) + w_C.reshape(3,1)
    normal_target_w = w_R_c @ normal.reshape(3,-1)
    p, n = point_target_w[:, mask], normal_target_w[:, mask]

    return p_pred, n_pred, p, n


# def SolveForPose(p_pred, n_pred, p):
#     """
#         Solve For Incremental Update Pose

#         Parameters
#         ----------
#         p_pred : ndarray of shape (3, -1)
#             Inlier tsdf point
#         n_pred : ndarray of shape (3, -1)
#             Inlier tsdf surface normal
#         p : ndarray of shape (3, -1)
#             Inlier depth image's point
#         Returns
#         -------
#         deltaT : ndarray of shape (4, 4)
#             Incremental updated pose matrix
#     """
#     n = p_pred.shape[1]

#     A = np.sum((np.concatenate([Vec2Skew(p.T).reshape(-1,3), 
#                             np.tile(np.eye(3),(n,1))], axis=1) * n_pred.reshape(-1,1)).reshape(n,3,6)
#                 , axis=1)
#     b = np.sum(n_pred * (p_pred - p), axis=0)
#     # A, b = [], []
#     # for i in range(n):
#     #     A.append(n_pred[:,i] @ np.concatenate([Vec2Skew(p[:,i]), np.eye(3)], axis=1))
#     #     b.append(n_pred[:,i] @ (p_pred[:,i] - p[:,i]))
#     # A, b = np.stack(A), np.stack(b)
#     sol = la.inv(A.T @ A) @ A.T @ b
#     deltaT = np.array([[      1,  sol[2], -sol[1], sol[3]],
#                        [-sol[2],       1,  sol[0], sol[4]],
#                        [ sol[1], -sol[0],       1, sol[5]],
#                        [      0,       0,       0,      1]])
#     return deltaT

def SolveForPose(p_pred, n_pred, p):
    """
        Solve For Incremental Update Pose
        Parameters
        ----------
        p_pred : ndarray of shape (3, -1)
            Inlier tsdf point
        n_pred : ndarray of shape (3, -1)
            Inlier tsdf surface normal
        p : ndarray of shape (3, -1)
            Inlier depth image's point
        Returns
        -------
        deltaT : ndarray of shape (4, 4)
            Incremental updated pose matrix
    """
    p_x = p[0]
    p_y = p[1]
    p_z = p[2]
    p_pred_x = p_pred[0]
    p_pred_y = p_pred[1]
    p_pred_z = p_pred[2]
    n_pred_x = n_pred[0]
    n_pred_y = n_pred[1]
    n_pred_z = n_pred[2]
    At1 = p_z * n_pred_y - p_y * n_pred_z
    At2 = -p_z * n_pred_x + p_x * n_pred_z
    At3 = p_y * n_pred_x - p_x * n_pred_y
    At4 = n_pred_x
    At5 = n_pred_y
    At6 = n_pred_z
    At = np.vstack((At1, At2, At3, At4, At5, At6))
    b = n_pred_x * (p_pred_x - p_x) + n_pred_y * (p_pred_y - p_y) + n_pred_z * (p_pred_z - p_z)
    x = np.linalg.pinv(At.T) @ b
    alpha = x[2]
    beta = x[0]
    gamma = x[1]
    tx = x[3]
    ty = x[4]
    tz = x[5]
    deltaT = np.array([[1., alpha, -gamma, tx],
                       [-alpha, 1., beta, ty],
                       [gamma, -beta, 1., tz],
                       [0., 0., 0., 1.]])
    return deltaT

def FuseTSDF(tsdf, tsdf_new):
    """
        FuseTSDF : Fusing 2 tsdfs

        Parameters
        ----------
        tsdf, tsdf_new : TSDFs
        Returns
        -------
        tsdf : TSDF
            Fused of tsdf and tsdf_new
    """
    mask = (tsdf.weight>0) | (tsdf_new.weight>0)
    tsdf.value[mask] = (tsdf.value[mask] * tsdf.weight[mask] + \
                        tsdf_new.value[mask] * tsdf_new.weight[mask])\
                        / (tsdf.weight[mask] + tsdf_new.weight[mask])
    tsdf.weight[mask] = tsdf.weight[mask] + tsdf_new.weight[mask]

    return tsdf


if __name__ == '__main__':
    DEPTH_FOLDER = 'depth_images'
    OUTPUT_FOLDER = 'results'
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    voxel_param = VoxelParams(3, 256)
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1]])
    depth_factor = 5000.
    n_iters = 3
    e_p = voxel_param.vox_size * 10.0
    e_n = np.cos(np.pi / 3.0)

    viz = False

    T_cur = np.eye(4)
    depth_file_list = open(os.path.join(DEPTH_FOLDER, 'filelist.list'), 'r').read().split('\n')
    depth_img = ProcessDepthImage(os.path.join(DEPTH_FOLDER, depth_file_list[0]), depth_factor)
    tsdf = CreateTSDF(depth_img, T_cur, voxel_param, K)
    SaveTSDFtoMesh('%s/mesh_initial.ply' % OUTPUT_FOLDER, tsdf, viz)
    # visualize_pose(None, None, tsdf)

    rays = ImageRays(K, voxel_param, depth_img.shape)
    for i_frame in range(1, len(depth_file_list)-1):
        print('process frame ', i_frame)

        point_pred, valid_rays = rays.cast(T_cur, voxel_param, tsdf)
        SavePointDepth('%s/pd_%02d.ply' % (OUTPUT_FOLDER, i_frame), point_pred, valid_rays, viz)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'depth_pred_%02d.png'%(i_frame)), (point_pred[2]/3*255).astype('int'))
        # visualize_pose(point_pred, None, tsdf)

        normal_pred = -ComputeTSDFNormal(point_pred, tsdf, voxel_param)
        SavePointNormal('%s/pn_%02d.ply' % (OUTPUT_FOLDER, i_frame), point_pred, normal_pred, valid_rays, viz)

        depth_img = ProcessDepthImage(os.path.join(DEPTH_FOLDER, depth_file_list[i_frame]), depth_factor)
        point, normal = Get3D(depth_img, K)

        for i in range(n_iters):
            p_pred, n_pred, p, n = FindCorrespondence(T_cur, point_pred, normal_pred,
                                                      point, normal, valid_rays, K, e_p, e_n)
            # visualize_pose(point_pred, point, None)
            # visualize_pose(p_pred, p, None)
            deltaT = SolveForPose(p_pred, n_pred, p)

            # Update pose
            T_cur = deltaT @ T_cur
            u, s, vh = np.linalg.svd(T_cur[:3, :3])
            R = u @ vh
            R *= np.linalg.det(R)
            T_cur[:3, :3] = R

        tsdf_new = CreateTSDF(depth_img, T_cur, voxel_param, K)
        # SaveTSDFtoMesh('%s/mesh_%02d_new.ply' % (OUTPUT_FOLDER, i_frame), tsdf_new, viz)
        tsdf = FuseTSDF(tsdf, tsdf_new)
        SaveTSDFtoMesh('%s/mesh_%02d.ply' % (OUTPUT_FOLDER, i_frame), tsdf, viz)

    SaveTSDFtoMesh('%s/mesh_final.ply' % OUTPUT_FOLDER, tsdf, viz)



