import os
import cv2
import argparse
import pickle
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

    pickle.dump(track, open('track.pkl', 'wb'))