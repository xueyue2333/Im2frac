import open3d as o3d
import numpy as np
import os


path = r"C:\Users\a\Desktop\all_ply\2"

for i in os.listdir(path):
    now = os.path.join(path, i)

    pcd = o3d.io.read_point_cloud(now)

    point = np.array(pcd.points)
    color = np.array(pcd.colors)
    normal = -np.array(pcd.normals)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(point)
    out.colors = o3d.utility.Vector3dVector(color)
    out.normals = o3d.utility.Vector3dVector(normal)

    o3d.io.write_point_cloud(now, out)
