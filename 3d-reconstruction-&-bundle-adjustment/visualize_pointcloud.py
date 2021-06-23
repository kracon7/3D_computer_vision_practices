import open3d as o3d 
from scipy.spatial.transform import Rotation as R
import numpy as np

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    camera = o3d.io.read_point_cloud("./output/cameras_4.ply")
    pcd = o3d.io.read_point_cloud("./output/points_4.ply")
    o3d.visualization.draw_geometries([camera, pcd])

    camera = o3d.io.read_point_cloud("./output/cameras_5.ply")
    pcd = o3d.io.read_point_cloud("./output/points_5.ply")
    o3d.visualization.draw_geometries([camera, pcd])

    camera = o3d.io.read_point_cloud("./output/cameras_6.ply")
    pcd = o3d.io.read_point_cloud("./output/points_6.ply")
    o3d.visualization.draw_geometries([camera, pcd])    
    # T = np.eye(4)
    # m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
    # m.transform(T)

    # T = np.eye(4)
    # T[:3,:3] = R.from_euler('x', 45, degrees=True).as_matrix()
    # T[:3, 3] = np.array([1,0,0])
    # print(T)
    # n = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # n.transform(T)

    # o3d.visualization.draw_geometries([m, n])