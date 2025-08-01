import numpy as np
import open3d as o3d


path = r"C:\Users\a\Desktop\cloud_mountain_4.8_mvsnet_4.8.ply"

ply = o3d.io.read_point_cloud(path)
print(o3d.io.write_point_cloud(path.replace("ply", "pcd"), ply))
