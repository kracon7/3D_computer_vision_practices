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
from camera_pose import draw_all
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment
import pickle
import matplotlib.pyplot as plt

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

    # load feature track
    track = pickle.load(open('track.pkl', 'rb'))

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

    ransac_n_iter = 500
    ransac_thr = 0.002
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
            # get correspoinding index
            mask_index = np.asarray(np.nonzero(mask)[0])

            # Triangulate points
            missing_X = Triangulation(P[i], P[j], track_i[mask], track_j[mask])
            missing_X = Triangulation_nl(missing_X, P[i], P[j], track_i[mask], track_j[mask])

            # Filter out points based on cheirality
            valid_index = EvaluateCheirality(P[i], P[j], missing_X)

            # Update 3D points
            X[mask_index[valid_index]] = missing_X[valid_index]

    visualize = 1
    if visualize:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        draw_all(ax, P, X)
        plt.show()