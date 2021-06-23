import cv2
import numpy as np
from scipy.spatial import cKDTree

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

    x1, x2, ind1 = [], [], []
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
                    ind1.append(i)
    x1 = np.stack(x1)
    x2 = np.stack(x2)
    ind1 = np.array(ind1)
    return x1, x2, ind1


def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """

    n = x1.shape[0]

    A=[];
    for i in range(n):
        ux, uy, vx, vy = x1[i, 0], x1[i, 1], x2[i, 0], x2[i, 1]
        A.append([ux*vx, uy*vx, vx, ux*vy, uy* vy, vy, ux, uy, 1])
        
    A = np.stack(A)

    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3,3)

    U, _, Vh = np.linalg.svd(E)
    D = np.eye(3)
    D[2,2] = 0
    E = U @ D @ Vh

    return E


def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    n = x1.shape[0]
    max_inlier = 0
    inlier = None
    E = np.eye(3)

    # homogeneous coordinates
    homo_x1 = np.insert(x1, 2, 1, axis=1)
    homo_x2 = np.insert(x2, 2, 1, axis=1)

    for n_step in range(ransac_n_iter):
        sample_idx = np.random.choice(n, 8) 
        sampled_x1 = x1[sample_idx]
        sampled_x2 = x2[sample_idx]

        estimated_E = EstimateE(sampled_x1, sampled_x2)

        error = np.abs(np.diag(homo_x1 @ estimated_E.T @ homo_x2.T))

        num_inlier = np.sum(error < ransac_thr)

        if num_inlier > max_inlier:
            max_inlier = num_inlier
            E = estimated_E
            inlier = np.array(np.nonzero(error < ransac_thr)).reshape(-1)

    return E, inlier


def cvt_keypoint(kp):
    '''
    convert keypoint result to numpy array
    '''
    N = len(kp)
    locations = np.zeros((N,2))
    for i in range(N):
        locations[i, :] = np.array(kp[i].pt)
    return locations


def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    N = Im.shape[0]
    
    loc_list, des_list, num_index = [], [], [0]
    sift = cv2.xfeatures2d.SIFT_create()
    
    print('Extract SIFT features...')
    for i in range(N):

        # compute keypoint and descriptors
        kp, des = sift.detectAndCompute(Im[i], None)
        # convert keypoint to numpy array
        loc = cvt_keypoint(kp)

        loc_list.append(loc)
        des_list.append(des)
        num_index.append(num_index[-1] + loc.shape[0])

        print('image %d, found %d features'%(i, loc.shape[0]))

    track = np.empty((N, 0, 2))
    for i in range(N - 1):
        print('Build track %d....'%(i))

        nft = loc_list[i].shape[0]
        track_i = -1 * np.ones((N, nft, 2))

        loc1 = loc_list[i]
        des1 = des_list[i]

        for j in range(i+1, N):
            
            loc2 = loc_list[j]
            des2 = des_list[j]

            x1, x2, ind1 = MatchSIFT(loc1, des1, loc2, des2)
            print('Found %d matched pairs between image %d and %d'%(x1.shape[0], i, j))

            # normalize coordinate by inv(K)
            norm_coord_1 = np.insert(x1, 2, 1, axis=1) @ np.linalg.inv(K).T
            norm_coord_2 = np.insert(x2, 2, 1, axis=1) @ np.linalg.inv(K).T

            norm_coord_1 = norm_coord_1[:, :2]
            norm_coord_2 = norm_coord_2[:, :2]

            E, inlier = EstimateE_RANSAC(norm_coord_1, norm_coord_2, 500, 0.003)
            print('%d matched pairs remains after essential matrix estimation'%(inlier.shape[0]))

            track_index = ind1[inlier]

            track_i[i, track_index, :] = norm_coord_1[inlier]
            track_i[j, track_index, :] = norm_coord_2[inlier]

        # filter features in ith image with no matches
        mask = np.sum(track_i[i], axis=1) != -2
        track_i = track_i[:, mask, :]
        print('Adding %d feature matches from image %d into track'%(track_i.shape[1], i))
        track = np.concatenate([track, track_i], axis=1)

    return track