import open3d as o3d
import numpy as np
import cv2
import os


def process_and_save_image(file_path,img_path):
    point_cloud = o3d.io.read_point_cloud("mvsnet_3.ply")
    matrix_data = np.genfromtxt(file_path, delimiter=' ', skip_header=1, max_rows=4)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 获取所需行
        line_number = 7
        line_number2 = 8
        desired_line = lines[line_number]
        desired_line2 = lines[line_number2]

        # 分割数据行
        values = desired_line.split()
        values2 = desired_line2.split()

        # 读取数值
        desired_value = values[0]
        w = values[2]
        y = values2[2]

    # 定义参数
    # 获取图像的像素大小（高度和宽度）
    img = cv2.imread(img_path)
    height, width, channels = img.shape

    '''
    width, height = int(2 * float(w)), int(2 * float(y))
    print(width,height)
    '''
    
    fx, fy, cx, cy = float(desired_value), float(desired_value), float(width / 2 - 0.5), float(height / 2 - 0.5)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # 初始化并赋值
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = np.array(matrix_data)
    param.intrinsic = intrinsic

    # 可视化点云
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=width, height=height)
    ctr = vis.get_view_control()
    vis.add_geometry(point_cloud)
    ctr.convert_from_pinhole_camera_parameters(param)
    # vis.run()

    # 保存图片
    file_number = os.path.splitext(os.path.basename(file_path))[0]
    img_name = f"outcrop\getImages\\{file_number}.jpg"

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(img_name)
    vis.destroy_window()


img_folder_path = r"outcrop\images"
txt_folder_path = r"outcrop\cams"

for img_file_name in os.listdir(img_folder_path):
    # 构造图像文件的完整路径
    img_path = os.path.join(img_folder_path, img_file_name)

    # 构造对应的txt文件的完整路径
    txt_file_name = img_file_name.replace('.jpg', '_cam.txt')
    file_path = os.path.join(txt_folder_path, txt_file_name)

    # 检查对应的txt文件是否存在
    if os.path.exists(file_path):
        process_and_save_image(file_path, img_path)





