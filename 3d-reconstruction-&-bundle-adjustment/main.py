import os
import cv2
import argparse
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='im')
    args = parser.parse_args()

    K = np.asarray([
        [650, 0, 640],
        [0, 650, 360],
        [0, 0, 1]
    ])
    
    # Load input images
    img_list = os.listdir(args.img_dir)
    img_list.sort()
    num_images = len(img_list)
    im_shape = cv2.imread(os.path.join(args.img_dir, img_list[0])).shape
    h_im = im_shape[0]
    w_im = im_shape[1]

    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        im = cv2.imread(os.path.join(args.img_dir, img_list[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i,:,:,:] = im

    # Build feature track
    track = BuildFeatureTrack(Im, K)

    track1 = track[0,:,:]
    track2 = track[1,:,:]

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track1, track2)

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))
    # Set first two camera poses
    P[0] = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]])
    P[1] = np.hstack([R, -(R@C).reshape((3,1))])

    ransac_n_iter = 1000
    ransac_thr = 0.005
    for i in range(2, num_images):
        # Estimate new camera pose
        X_mask = np.logical_and( np.logical_and(X[:,0]!=-1, X[:,1]!=-1), X[:,2]!=-1)

        track_i = track[i,:,:]
        track_mask_i = np.logical_and(track_i[:,0]!=-1, track_i[:,1]!=-1)

        mask = np.logical_and(X_mask, track_mask_i)
        R, C, inlier = PnP_RANSAC(X[mask], track_i[mask], ransac_n_iter, ransac_thr)
        R, C = PnP_nl(R, C, X[mask], track_i[mask])

        # Add new camera pose to the set
        P[i] = np.hstack([R, -(R@C).reshape((3,1))])

        for j in range(i):
            # Fine new points to reconstruct
            track_j = track[j,:,:]
            track_mask_j = np.logical_and(track_j[:,0]!=-1, track_j[:,1]!=-1)

            # get mask for points that exist in both track_i and track_j but not in X
            mask = np.logical_and( np.logical_and(track_mask_i, track_mask_j), ~X_mask)
            # get correspoinding index in X and track
            mask_index = np.asarray(np.nonzero(mask)[0])

            # Triangulate points
            print('Running linear triangulation between image %d and %d'%(i, j))
            missing_X = Triangulation(P[i], P[j], track_i[mask], track_j[mask])
            missing_X = Triangulation_nl(missing_X, P[i], P[j], track_i[mask], track_j[mask])

            # Filter out points based on cheirality
            valid_index = EvaluateCheirality(P[i], P[j], missing_X)

            # Update 3D points
            X[mask_index[valid_index]] = missing_X[valid_index]
        
        # Run bundle adjustment
        valid_ind = X[:, 0] != -1
        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        P_new, X_new = RunBundleAdjustment(P[:i + 1, :, :], X_ba, track_ba)
        P[:i + 1, :, :] = P_new
        X[valid_ind, :] = X_new

        # X_new = X[valid_ind, :]

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d.T
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0],1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j,:,:].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
            uv = x[mask_valid,:] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(w_im), Im[j,:,:,k].astype(float)/255, kx=1, ky=1)
                colors[mask_valid, k] = interp_fun(uv[:,1], uv[:,0], grid=False)

        ind = np.sqrt(np.sum(X_ba ** 2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
        pcd.colors = o3d.utility.Vector3dVector(colors[ind])
        o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)