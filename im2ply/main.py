import os
import time
import open3d as o3d
from PIL import Image
import shutil


init_path = r'..\pytorch-mvsnet-master\namedData\cloud_mountain\4.8'

print("开始稀疏重建")
os.system('python colmap方法.py --base_dir {} --max_num_features {}'.format(init_path, 8192))
print("开始相机参数估计")
os.system('python colmap2mvsnet.py --dense_folder {}/dense'.format(init_path))
print("调图片信息")
os.system("python change_name.py --base_dir {}".format(init_path))
print("开始稠密重建")
os.system(r'python eval.py --filepath {}\dense --test_name {} --outdir {}'.format(init_path, "_".join(init_path.split("\\")[-2:]), r"outputs"))

pcd = o3d.io.read_point_cloud(r'outputs\{}_mvsnet_{}.ply'.format("_".join(init_path.split("\\")[-2:]), "_".join(init_path.split("\\")[-1:])))
o3d.visualization.draw_geometries([pcd], window_name="原始点云",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)
