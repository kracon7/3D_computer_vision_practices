import os
import cv2
import pickle
import argparse
import numpy as np
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    x1, x2 = [], []
    ratio = 0.7
    tree1, tree2 = cKDTree(des1), cKDTree(des2)

    N = loc1.shape[0]

    for i in range(N):
        ft_1 = des1[i]
        dd, ii = tree2.query([ft_1], k=2, n_jobs=-1)
        if dd[0,0] / dd[0,1] < ratio:
            # the correspoinding index of matched feature in des2 from des1
            idx2 = ii[0, 0]
            
            # query back from feature 2 to tree1
            ft_2 = des2[idx2]
            dd, ii = tree1.query([ft_2], k=2, n_jobs=-1)
            if dd[0,0] / dd[0,1] < ratio:
                if ii[0, 0] == i:
                    x1.append(loc1[i])
                    x2.append(loc2[idx2])

    x1 = np.stack(x1)
    x2 = np.stack(x2)
    return x1, x2

def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    max_inlier = 0
    res = np.eye(3)
    inlier = None

    N =x1.shape[0]
    # homogeneous coordinates for x1 and x2
    homo_x1 = np.insert(x1, 2, 1, axis=1)
    homo_x2 = np.insert(x2, 2, 1, axis=1)

    for n_step in range(ransac_n_iter):
        # random sample four correspondences
        sample_idx = np.random.choice(N, 4)
        # construct homography computation matrix A
        A = []
        for idx in sample_idx:
            p1, p2 = x1[idx], x2[idx]
            a = np.array([[p1[0], p1[1], 1, 0,     0,     0, -p1[0]*p2[0], -p1[1]*p2[0], -p2[0]],
                          [0,     0,     0, p1[0], p1[1], 1, -p1[0]*p2[1], -p1[1]*p2[1], -p2[1]]])
            A.append(a)
        # solve Ax = 0
        A = np.vstack(A)
        u, s, v = np.linalg.svd(A)
        h = v[-1,:]
        H = h.reshape(3,3) / np.linalg.norm(h)

        # compute inliers
        # map x1 homogeneous coordinates by H
        mapped_x1 = np.dot(H, homo_x1.T).T
        # normalize mapped coordinates to be homogeneous (last element == 1)
        norm_mapped_x1 = (mapped_x1.T / mapped_x1[:, 2]).T
        error = np.linalg.norm(norm_mapped_x1 - homo_x2, axis=1)
        num_inlier = np.sum(error < ransac_thr)
        if num_inlier > max_inlier:
            max_inlier = num_inlier
            res = H
            inlier = np.array(np.nonzero(error < ransac_thr))
    return res, inlier.reshape(-1)


def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """
    H = H/H[2,2]
    R = np.linalg.inv(K) @ H @ K
    detR = np.linalg.det(R)
    R = 1 / np.cbrt(detR) * R
    # R should be rotational matrix with determinant of 1
    assert np.linalg.det(R) - 1.0 < 1e-4
    return R


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Wc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """
    f = K[0,0]
    d_theta = 2*np.pi / Wc
    theta = np.arange(0, 2*np.pi, d_theta)
    y = np.arange(0, Hc) - Hc/2
    xx, yy = np.meshgrid(theta, y)
    p = np.stack([f*np.sin(xx), yy, f*np.cos(xx)], axis=-1)
    assert p.shape == (Hc, Wc, 3)
    return p    


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """
    Hc, Wc, _ = p.shape
    u = np.zeros((Hc, Wc, 2)).astype('int')
    mask = np.zeros((Hc, Wc))
    
    
    p = p.reshape(-1, 3)
    p = p @ R.T
    pixel = p / p[:, -1:] @ K.T
    u = pixel[:, [1,0]].reshape(Hc, Wc, 2).astype('int')
    
    front_idx = p[:, 2] > 0
    
    inbound_idx = np.logical_and(
                np.logical_and(pixel[:,0] > 0, pixel[:,0] < W),
                np.logical_and(pixel[:,1] > 0, pixel[:,1] < H))
    valid_idx = np.logical_and(front_idx, inbound_idx)
    mask = valid_idx.reshape(Hc, Wc).astype('int')

    return u, mask
            

def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """
    Hc, Wc = mask_i.shape
    canvas_i = np.zeros((Hc, Wc, 3))
    canvas_i = canvas_i.reshape(-1, 3)
    u = u.reshape(-1, 2)
    mask_i = mask_i.reshape(-1)
    pixels = u[mask_i > 0]
    canvas_i[mask_i > 0] = image_i[pixels[:,0], pixels[:,1], :]
    return canvas_i.reshape(Hc, Wc, 3)

    
def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """
    canvas[mask_i > 0] = canvas_i[mask_i > 0]
    return canvas


def cvt_keypoint(kp):
    '''
    convert keypoint result to numpy array
    '''
    N = len(kp)
    locations = np.zeros((N,2))
    for i in range(N):
        locations[i, :] = np.array(kp[i].pt)
    return locations

def draw_sift_match(im1, im2, x1, x2):
    '''
    visualize the filtered sift feature matches
    '''
    m, n, _ = im1.shape
    I = np.concatenate([im1, im2], axis=1)
    plt.ion()
    fig, ax = plt.subplots(1,1)
    ax.imshow(I)
    # compute shifted x2 coordinates
    sx2 = x2.copy()
    sx2[:,0] += n
    ax.plot(x1[:,0], x1[:,1], 'g+')
    ax.plot(sx2[:,0], sx2[:,1], 'r+')
    ax.plot([x1[:,0], sx2[:,0]],[x1[:,1], sx2[:,1]], 'y')    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='images')
    args = parser.parse_args()

    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])

    # Read all images
    im_list = []
    flist = os.listdir(args.img_dir)
    flist.sort()
    for i in range(len(flist)):
        im_file = flist[i]
        im = cv2.imread(os.path.join(args.img_dir, im_file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    print('Running through images to compute relative rotations...')
    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
        I_prev = im_list[i]
        I_next = im_list[i+1]

        # Extract SIFT features
        sift = cv2.xfeatures2d.SIFT_create()
        # compute keypoint and descriptors
        kp1, des1 = sift.detectAndCompute(I_prev, None)
        kp2, des2 = sift.detectAndCompute(I_next, None)
        # convert keypoint to numpy array
        loc1 = cvt_keypoint(kp1)
        loc2 = cvt_keypoint(kp2)
        
        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # draw_sift_match(I_prev, I_next, x1, x2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # draw_sift_match(I_prev, I_next, x1[inlier], x2[inlier])

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)
        
        # Compute R_new (or R_i+1)
        R_new = rot_list[-1] @ R
        
        rot_list.append(R_new)

        print('Finished %i images'%(i))

    print('Dumping rot_list to pickle file')
    pickle.dump(rot_list, open('rotlist.pkl', 'wb'))

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]
    
    Hc = Him
    Wc = len(im_list) * Wim // 2
    
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure('HW1')
    plt.axis('off')
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        print('Done warping %dth image onto canvas'%(i))        
        plt.imshow(canvas)
        plt.pause(1)
        plt.savefig('output_{}.png'.format(i+1), dpi=600, bbox_inches = 'tight', pad_inches = 0)