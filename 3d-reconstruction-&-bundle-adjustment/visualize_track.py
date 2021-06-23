import os
import cv2
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
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]
    ])
    num_images = 6
    h_im = 540
    w_im = 960

    # Load input images
    Im = []
    for i in range(num_images):
        im_file = 'im/image{:07d}.jpg'.format(i + 1)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im.append(im)

    # load feature track
    track = pickle.load(open('track.pkl', 'rb'))

    F = track.shape[1]
    N = 10

    sample_idx = np.random.choice(F, N)
    track = track[:, sample_idx, :]

    mask = np.zeros((num_images, N, 1))
    for i in range(num_images):
        for j in range(N):
            if track[i,j,0] != -1 and track[i,j,1] != -1:
                mask[i, j] = 1
    start_idx = [np.min(np.nonzero(mask[:,i])[0]) for i in range(mask.shape[1]) ]

    cmap = plt.cm.get_cmap('Spectral')
    colors = [cmap(i)    for i in np.linspace(0, 1, N)]

    for j in range(N):
        color = colors[j]

        # find start pixel
        start_ray = np.array([track[start_idx[j], j, 0], track[start_idx[j], j, 1], 1])
        start_pixel = (K @ start_ray)[:2]
        start_pixel[1] += start_idx[j] * h_im

        for i in range(num_images):    
            if track[i,j,0] != -1 and track[i,j,1] != -1:
                ray = np.array([track[i,j,0], track[i,j,1], 1])
                pixel = (K @ ray)[:2]
                pixel[1] += i * h_im

                plt.plot(pixel[0], pixel[1], color=color, marker='x')
                plt.plot([start_pixel[0], pixel[0]],[start_pixel[1], pixel[1]], color=color)

                # if pixel[0] > (num_images-1) * h_im:
                #     print('image # %d, feature # %d'%(i, j), K@ray)

    canvas = np.concatenate(Im, axis=0)

    plt.imshow(canvas)
    plt.show()