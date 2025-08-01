import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import open3d as o3d


def process_and_save_image(file_path, img_path, ply_path, init_path):
    if not os.path.exists(f"{init_path}\dense\getImages"):
        os.makedirs(f"{init_path}\dense\getImages")

    point_cloud = o3d.io.read_point_cloud(ply_path)
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
    img_name = f"{init_path}\dense\getImages\\{file_number}.jpg"

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(img_name)
    vis.destroy_window()


def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    _, white_mask = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY_INV)
    image1 = cv2.bitwise_and(image1, image1, mask=white_mask)
    image2 = cv2.bitwise_and(image2, image2, mask=white_mask)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

 
def classify_hist_with_split(image1, image2):
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


def cir_acc(init_path):
    temp = init_path.split("\\")[-1]
    ply_path = os.path.join("outputs", f"{temp[1:]}_mvsnet_{temp}.ply")

    img_folder_path = os.path.join(init_path, "dense/images")
    txt_folder_path = os.path.join(init_path, "dense/cams")

    for img_file_name in os.listdir(img_folder_path):
        # 构造图像文件的完整路径
        img_path = os.path.join(img_folder_path, img_file_name)

        # 构造对应的txt文件的完整路径
        txt_file_name = img_file_name.replace('.jpg', '_cam.txt')
        file_path = os.path.join(txt_folder_path, txt_file_name)

        # 检查对应的txt文件是否存在
        if os.path.exists(file_path):
            process_and_save_image(file_path, img_path, ply_path, init_path)

    # 定义两个文件夹的路径
    img_folder1 = os.path.join(init_path, "dense/images")
    img_folder2 = os.path.join(init_path, "dense/getImages")
    result_list = []
    # 遍历第一个文件夹中的图片
    for img_name1 in os.listdir(img_folder1):
        img_path1 = os.path.join(img_folder1, img_name1)

        # 构造第二个文件夹中对应的图片路径
        img_name2 = img_name1.replace('.jpg', '_cam.jpg')
        img_path2 = os.path.join(img_folder2, img_name2)

        # 检查第二个文件夹中是否存在对应的图片
        if os.path.exists(img_path2):
            # 调用函数处理对应的图片
            img1 = cv2.imread(img_path1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2 = cv2.imread(img_path2)
            img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)

            n4 = classify_hist_with_split(img1, img2)
            square = n4
            result_list.append(square)

    average = sum(result_list) / len(result_list)
    return average


if __name__ == '__main__':
    path = "real"
    for i in os.listdir(path):
        if "max_error" in i:
            print("parameter: {}".format(i), end="---")
            now = cir_acc(os.path.join(path, i))
            print("average: {}".format(now))
