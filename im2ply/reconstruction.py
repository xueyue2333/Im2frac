import open3d as o3d
import numpy as np
import regiongrowing as reg
import random
import math
import copy
import os

"""
a = np.asarray(pcd.points)
b = np.asarray(pcd.colors)
a[0]
b[0]
"""

# ------------------------------读取点云---------------------------------------
pcd = o3d.io.read_point_cloud(r"outputs\mvsnet_tupian_shipin_tanjiahe_qian.ply")
# pcd = pcd.voxel_down_sample(voxel_size=0.1)
print(len(np.array(pcd.points)))
o3d.visualization.draw_geometries([pcd], window_name="原始点云",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)
# ------------------------------区域生长---------------------------------------

rg = reg.RegionGrowing(pcd,
                       min_pts_per_cluster=300,  # 每个聚类的最小点数
                       max_pts_per_cluster=100000,  # 每个聚类的最大点数
                       neighbour_number=1500,  # 邻域搜索点数
                       theta_threshold=20,  # 平滑阈值（角度制）
                       curvature_threshold=0.06)  # 曲率阈值

# rg = reg.RegionGrowing(pcd)
# ---------------------------聚类结果分类保存----------------------------------
indices = rg.extract()
print("聚类个数为", len(indices))
segment = []  # 存储分割结果的容器
for i in range(len(indices)):
    ind = indices[i]
    clusters_cloud = pcd.select_by_index(ind)
    r_color = np.random.uniform(0, 1, (1, 3))  # 分类点云随机赋色
    clusters_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
    segment.append(clusters_cloud)
segment.append(pcd)

o3d.io.write_point_cloud(r"D:\A-Fkf\1.ply", pcd)

path = r"D:\A-Fkf\MVSNet_pytorch-master\fractures\1_1"
if not os.path.exists(path):
    os.makedirs(path)

for i in range(len(segment) - 1):
    o3d.io.write_point_cloud(os.path.join(path, r"{}.ply".format(i)),
                             segment[i])

# -----------------------------结果可视化------------------------------------
o3d.visualization.draw_geometries(segment, window_name="区域生长分割",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=True)

# ---------------------------RANSAC拟合平面----------------------------------

XX = []
for i in range(len(segment)):

    points = np.asarray(segment[i].points)
    A = np.ones((len(points), 3))
    b = np.ones((len(points), 1))

    for j in range(len(points)):
        A[j, 0] = points[j][0]
        A[j, 1] = points[j][1]
        b[j, 0] = points[j][2]

    for t in range(500):
        n = []
        plane = []
        a = np.ones((500, 3))
        bb = np.ones((500, 1))

        for p in range(500):
            n.append(random.randint(0, len(points) - 1))
            a[p, 0] = A[n[p]][0]
            a[p, 1] = points[n[p]][1]
            bb[p, 0] = points[n[p]][2]
        a_T = a.T
        a1 = np.dot(a_T, a)
        a2 = np.linalg.inv(a1)
        a3 = np.dot(a2, a_T)
        X = np.dot(a3, bb)

        R0 = 0
        R1 = 1e10
        for q in range(len(points)):
            R0 = R0 + (X[0, 0] * A[q, 0] + X[1, 0] * A[q, 1] + X[2, 0] - b[q, 0]) ** 2

        if t == 1:
            X_better = np.dot(a3, bb)
        elif R0 < R1:
            X_better = X
        R1 = R0

    XX.append(X_better)

# --------------------------------平面合并------------------------------------
dis = np.zeros((len(segment), len(segment)))
angel = np.zeros((len(segment), len(segment)))
for i in range(len(segment)):
    for j in range(i, len(segment)):
        if i == j:
            dis[i][j] = 0
        else:
            n1 = []
            dis1 = 0
            for p in range(500):
                points = np.asarray(segment[i].points)
                n1.append(random.randint(0, len(np.asarray(segment[i].points))))
                dis1 = dis1 + abs(XX[j][0] * points[n1[-1] - 1, 0] + XX[j][1] * \
                                  points[n1[-1] - 1, 1] + points[n1[-1] - 1, 2] + XX[j][2]) / \
                       math.sqrt(XX[j][0] ** 2 + XX[j][1] ** 2 + 1)
            dis1 = dis1 / 500
            dis[i][j] = dis1
        angel1 = 0
        if i == j:
            angel[i][j] = 0
        else:
            angel1 = 180 / math.pi * math.acos((XX[i][0] * XX[j][0] + XX[i][1] * XX[j][1] + 1) / \
                                               (math.sqrt(XX[i][0] ** 2 + XX[i][1] ** 2 + 1) * math.sqrt(
                                                   XX[j][0] ** 2 + XX[j][1] ** 2 + 1)))
            angel[i][j] = angel1

plane_same = []
for i in range(len(segment)):
    for j in range(i, len(segment)):
        if dis[i][j] != 0 and dis[i][j] < 2.5 and angel[i][j] != 0 and (angel[i][j] < 50 or angel[i][j] > 130):
            plane_same.append([i, j])

myset = set()
mylist1 = []
mylist2 = []
for i in range(len(plane_same)):
    myset.add(plane_same[i][0])
mylist = list(myset)
j = 0
mylist1 = []
for i in range(len(mylist)):
    myset1 = set()
    while True:
        if j == len(plane_same):
            break
        elif mylist[i] == plane_same[j][0]:
            myset1.add(plane_same[j][0])
            myset1.add(plane_same[j][1])
            j += 1
        else:
            break
    mylist1.append(list(myset1))
    mylist2.append(myset1)

num = []
for i in range(len(mylist1)):
    num.append(i)

i = 0
if mylist1:
    while True:
        j = 0
        while True:
            k = i + 1
            print(k)
            print('---------------------------------')
            while True:
                mylist3 = []
                if k == len(mylist1):
                    break
                elif mylist1[i][j] in mylist2[k]:
                    mylist2[i].update(mylist1[k])
                    mylist3 = list(mylist2[i])
                    for n in range(len(mylist3)):
                        if mylist3[n] not in mylist1[i]:
                            mylist1[i].append(mylist3[n])
                    mylist2.pop(k)
                    mylist1.pop(k)
                    print(mylist1)
                else:
                    k += 1
                    print(k)
                    print('++++++++++++++++++++++++++++++')
            j += 1
            if j == len(mylist1[i]):
                break
        i += 1
        if i == len(mylist1) - 1:
            break

# -------------------------------显示合并后平面-------------------------------
indices1 = copy.deepcopy(indices)
num1 = []
for i in range(len(mylist1)):
    for j in range(1, len(mylist1[i])):
        indices1[mylist1[i][0]] = indices1[mylist1[i][j]] + indices1[mylist1[i][0]]
        num1.append(mylist1[i][j])

indices2 = []
for i in range(len(indices1)):
    if i not in num1:
        indices2.append(indices1[i])

segment1 = []  # 存储分割结果的容器
for i in range(len(indices2)):
    ind1 = indices2[i]
    clusters_cloud1 = pcd.select_by_index(ind1)
    r_color = np.random.uniform(0, 1, (1, 3))  # 分类点云随机赋色
    clusters_cloud1.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
    segment1.append(clusters_cloud1)

o3d.visualization.draw_geometries(segment1, window_name="平面拾取结果",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)
