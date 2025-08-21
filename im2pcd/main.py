import os
import open3d as o3d
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--init_path", type=str, required=False, help="image path",
                    default=r'..\im2pcd\example_data\data')
args = parser.parse_args()

init_path = args.init_path

print("Start sparse reconstruction")
os.system('python colmap方法.py --base_dir {} --max_num_features {}'.format(init_path, 8192))
print("Start estimating the camera parameters")
os.system('python colmap2mvsnet.py --dense_folder {}/dense'.format(init_path))
print("Adjust the image information")
os.system("python change_name.py --base_dir {}".format(init_path))
print("Start dense reconstruction")
os.system(r'python eval.py --filepath {}\dense --test_name {} --outdir {}'.format(init_path, "_".join(init_path.split("\\")[-2:]), r"outputs"))

data_path = r'outputs\{}_mvsnet_{}.ply'.format("_".join(init_path.split("\\")[-2:]), "_".join(init_path.split("\\")[-1:]))
ply = o3d.io.read_point_cloud(data_path)

o3d.visualization.draw_geometries([ply], window_name="Point cloud",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)

o3d.io.write_point_cloud(data_path.replace("ply", "pcd"), ply)
