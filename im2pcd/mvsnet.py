import os
import time
import open3d as o3d
from PIL import Image


path = r'D:\A-Fkf\MVSNet_pytorch-master\A-NewData\tanjiahe_shouji_qian'

a = time.time()
print("开始稀疏重建")
os.system('python colmap方法.py --base_dir {}'.format(path))
print("开始相机参数估计")
os.system('python colmap2mvsnet.py --dense_folder {}/dense'.format(path))
print("调图片信息")
os.system("python change_name.py --base_dir {}".format(path))
print("开始稠密重建")
os.system(r'python eval.py --filepath {}\dense'.format(path))
b = time.time()
print("总耗时：{}".format(b - a))


pcd = o3d.io.read_point_cloud(r"D:\A-Fkf\MVSNet_pytorch-master\outputs\mvsnet_{}.ply".format(path.split("\\")[-1]))
o3d.visualization.draw_geometries([pcd], window_name="原始点云",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)
