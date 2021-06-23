import numpy.linalg as la
import scipy.io
from scipy import ndimage
from tsdf import *
from utils import *
import matplotlib.pyplot as plt
plt.ion()


class ImageRays:
    def __init__(self, K, voxel_param=VoxelParams(3, 256), im_size=np.array([480, 640])):
        """
            ImageRays : collection of geometric parameters of rays in an image

            Parameters
            ----------
            K : ndarray of shape (3, 3)
                Intrinsic parameters
            voxel_param : an instance of voxel parameter VoxelParams
            im_size: image size

            Class variables
            -------
            im_size : ndarray of value [H, W]
            rays_d : ndarray of shape (3, H, W)
                Direction of each pixel ray in an image with intrinsic K and size [H, W]
            lambda_step : ndarray (-1, )
                Depth of casted point along each ray direction
        """
        self.im_size = im_size
        h, w = im_size
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        points = np.stack([xx, yy], axis=2).reshape(-1,2)
        uv1 = np.insert(points, 2, 1, axis=1) @ la.inv(K).T
        # uv1 = np.linalg.inv(K) @ np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, h * w))
        self.rays_d = uv1 / np.linalg.norm(uv1, axis=1, keepdims=True)
        self.lambda_step = np.arange(voxel_param.vox_size, voxel_param.phy_len, voxel_param.vox_size)


    def cast(self, T, voxel_param, tsdf):
        """
            cast : ImageRays class' member function
                Collection of geometric parameters of rays in an image

            Parameters
            ----------
            T : ndarray of shape (4, 4)
                Transformation that brings camera to world coordinate
            voxel_param : an instance of voxel parameter VoxelParams
            tsdf : an instance of TSDF

            Returns
            -------
            point_pred : ndarray of shape (3, H, W)
                Point cloud from casting ray to tsdf
            valid : ndarray of shape (H, W)
                Mask to indicate which points are properly casted
        """
        n_lambda = self.lambda_step.shape[0]

        im_h, im_w = self.im_size
        point_pred, valid = np.zeros((im_h*im_w, 3)), np.zeros(im_h*im_w)

        w_R_c, w_C = T[:3, :3], T[:3, 3]

        # tsdf_value = voxel_param.trunc_thr * tsdf.value
        
        for i, ray in enumerate(self.rays_d):
            # positions of the points along this ray in camera frame
            points_c = ray.reshape(3,1) @ self.lambda_step.reshape(1, n_lambda)

            # transform the points into world frame
            points_w = (  T @ np.insert(points_c, 3, 1, axis=0)  )[:3]

            # coordinates of the points in tsdf
            coord = (points_w - voxel_param.voxel_origin.reshape(3,1)) \
                        / voxel_param.vox_size
            value = ndimage.map_coordinates(tsdf.value, coord, order=1)

            # if i%159==0:
            #     visualize_ray(points_w, point_pred, tsdf)
                # visualize(pt, tsdf)

            idx = np.where((value[:-1] * value[1:]) < 0)[0]
            if idx.shape[0] >= 2:
                k = idx[0]
                point_pred[i] = points_w[:,k] - value[k] / (value[k+1]-value[k]) * \
                                    (points_w[:,k+1]-points_w[:,k])
                valid[i] = 1

        # point_pred[(point_pred[:,2]>0) | (point_pred[:,2]<voxel_param.phy_len)] = 0
        point_pred = point_pred.reshape(im_h, im_w, 3).transpose(2,0,1)
        valid = valid.reshape(im_h, im_w)
        return point_pred, valid.astype('bool')



def visualize_ray(pt, point, tsdf):
    meshes = []
    if pt is not None:
        pt = pt.reshape(3,-1).T
        pcd_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pt))
        pcd_1.paint_uniform_color(np.array([1,0,0]))
        meshes.append(pcd_1)

    if point is not None:
        pcd_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point))
        pcd_2.paint_uniform_color(np.array([0,0,1]))
        meshes.append(pcd_2)

    if tsdf is not None:
        mask = tsdf.weight >0
        tsdf_point = np.vstack([tsdf.voxel_x[mask], tsdf.voxel_y[mask], tsdf.voxel_z[mask]]).T
        tsdf_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tsdf_point))
        tsdf_pcd.paint_uniform_color(np.array([0,1,0]))
        meshes.append(tsdf_pcd)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    meshes.append(mesh_frame)
    o3d.visualization.draw_geometries(meshes)

def visualize(pt, tsdf):
    meshes = []
    if pt is not None:
        pt = pt.reshape(3,-1).T
        pcd_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pt))
        pcd_1.paint_uniform_color(np.array([1,0,0]))
        meshes.append(pcd_1)

    if tsdf is not None:
        idx = np.where(tsdf.value<1)
        point = np.stack([idx[0], idx[1], idx[2]]).T
        tsdf_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point))
        tsdf_pcd.paint_uniform_color(np.array([0,1,0]))
        meshes.append(tsdf_pcd)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    meshes.append(mesh_frame)
    o3d.visualization.draw_geometries(meshes)