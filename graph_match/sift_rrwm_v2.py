from collections import defaultdict
from math import sqrt

import cv2
import os
import numpy as np
import torch  # pytorch backend
import pygmtools as pygm
import matplotlib.pyplot as plt  # for plotting
from matplotlib.patches import ConnectionPatch  # for plotting matching result
import networkx as nx  # for plotting graphs
from scipy.spatial import Delaunay
import itertools
import functools
from scipy.spatial import distance

pygm.BACKEND = 'pytorch'  # set default backend for pygmtools
_ = torch.manual_seed(1)  # fix random seed

big_img_path = './1235291h42cqan21nfrhcfe41ov8sms8/l02.png'
small_img_path = './1235291h42cqan21nfrhcfe41ov8sms8/l01.png'

# big_img_path = './1235291h42cqan21nfrhcfe41ov8sms8/D02.png'
# small_img_path = './1235291h42cqan21nfrhcfe41ov8sms8/D01.png'

big = cv2.imread(big_img_path)
big_h, big_w, _ = big.shape
small = cv2.imread(small_img_path)
small_h, small_w, _ = small.shape

# cv2.imshow("big", big)
# cv2.imshow("small", small)
# 创建SIFT特征检测器
sift = cv2.SIFT_create()
# 特征点提取与描述子生成
kp1, des1 = sift.detectAndCompute(small, None)
# print('kp1', kp1)
# for i in kp1:
#     print(i.response)
# print('des1', des1)
kp2, des2 = sift.detectAndCompute(big, None)

points1 = cv2.KeyPoint_convert(kp1)  # 将KeyPoint格式数据中的xy坐标提取出来。
points2 = cv2.KeyPoint_convert(kp2)  # 将KeyPoint格式数据中的xy坐标提取出来。


def xywh_2_x1y1x2y2(yolo_box, w, h):
    x_, y_, w_, h_ = yolo_box[1], yolo_box[2], yolo_box[3], yolo_box[4]
    x1 = w * x_ - 0.5 * w * w_
    x2 = w * x_ + 0.5 * w * w_
    y1 = h * y_ - 0.5 * h * h_
    y2 = h * y_ + 0.5 * h * h_

    return [x1, y1, x2, y2]


# 获取标注框
def get_boxes(img_path, img_w, img_h):
    label_path = os.path.splitext(img_path)[0] + '.txt'
    boxes = []
    with open(label_path, encoding='utf8') as f:
        for line in f.readlines():
            yolo_box = list(map(float, line.strip().split(' ')))
            boxes.append(xywh_2_x1y1x2y2(yolo_box, img_w, img_h))

    return boxes


big_boxes = get_boxes(big_img_path, big_w, big_h)
small_boxes = get_boxes(small_img_path, small_w, small_h)


def check_in_box(box, point):
    '''
    box : xyxy format
    point: x,y
    '''
    x, y = point
    if ((x >= box[0]) & (x <= box[2])) & ((y >= box[1]) & (y <= box[3])):
        return True
    else:
        return False


def filter_points(points, boxes, kp, des):
    # 获取每个框内的坐标点
    box_id_point_id_dic = defaultdict(list)
    for point_id, point in enumerate(points):
        for box_id, box in enumerate(boxes):
            if check_in_box(box, point):
                box_id_point_id_dic[box_id].append(point_id)
                # break

    # 每个框内留下离中心点最近的点
    filter_point_id_l = []
    ret_kp = []
    for box_id, point_id_list in box_id_point_id_dic.items():
        box = boxes[box_id]
        c_x = (box[0] + box[2]) / 2
        c_y = (box[1] + box[3]) / 2
        point_id_list = box_id_point_id_dic[box_id]
        dist = lambda point_id: np.linalg.norm(np.array([points[point_id]]) - np.array([c_x, c_y]))

        sorted_point_id_list = sorted(point_id_list, key=dist)
        nearest_point_id_list = sorted_point_id_list[:1]
        # nearest_point_id = min(point_id_list, key=dist)
        for nearest_point_id in nearest_point_id_list:
            filter_point_id_l.append(nearest_point_id)
            ret_kp.append(kp[nearest_point_id])

    return tuple(ret_kp), des[filter_point_id_l], points[filter_point_id_l]


print(f'len(points1): {len(points1)}')
kp1, des1, points1 = filter_points(points1, small_boxes, kp1, des1)
print(f'len(points1): {len(points1)}')
print(f'len(points2): {len(points2)}')
kp2, des2, points2 = filter_points(points2, big_boxes, kp2, des2)
print(f'len(points2): {len(points2)}')


def get_adjacency_matrix(points):
    n = len(points)  # 节点数量
    d = Delaunay(points)
    # 构建邻接矩阵A
    A = np.zeros((n, n))

    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
            # p1, p2 = pair
            # dis = sqrt(
            #     pow(int(points[p1][0]) - int(points[p2][0]), 2) + pow(int(points[p1][1]) - int(points[p2][1]), 2))
            # A[pair] = int(dis)
    return A


A1 = get_adjacency_matrix(points1)
A2 = get_adjacency_matrix(points2)
G1 = nx.from_numpy_array(A1)
G2 = nx.from_numpy_array(A2)

# Build affinity matrix
conn1, edge1 = pygm.utils.dense_to_sparse(torch.tensor(A1))
conn2, edge2 = pygm.utils.dense_to_sparse(torch.tensor(A2))

gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001)  # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, torch.tensor([len(A1)]), None,
                             torch.tensor([len(A2)]), None, edge_aff_fn=gaussian_aff)
n1 = torch.tensor([len(A1)])
n2 = torch.tensor([len(A2)])
X = pygm.rrwm(K, n1, n2)
X = pygm.hungarian(X)

plt.figure(figsize=(20, 10), dpi=100)
plt.suptitle(f'RRWM Matching Result ')

small_img = cv2.imread(small_img_path)
small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
ax1 = plt.subplot(1, 2, 1)
plt.imshow(small_img)

plt.title('Subgraph 1')
plt.gca().margins(0.4)
nx.draw_networkx(G1, pos=points1, with_labels=False, node_size=100)

big_img = cv2.imread(big_img_path)
big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
ax2 = plt.subplot(1, 2, 2)
plt.imshow(big_img)
plt.title('Graph 2')
nx.draw_networkx(G2, pos=points2, with_labels=False, node_size=100)


def get_distance_matrix(graph):
    node_list = graph.nodes()
    node_num = len(node_list)
    distance_matrix = np.zeros((node_num, node_num))
    for pair in itertools.combinations(node_list, 2):
        distance = nx.shortest_path_length(graph, source=pair[0], target=pair[1])
        distance_matrix[pair] = distance
        distance_matrix[pair[::-1]] = distance

    return distance_matrix


# 获取距离矩阵
distance_matrix_1 = get_distance_matrix(G1)
'''
[[0. 1. 1. 2. 1.]
 [1. 0. 1. 1. 1.]
 [1. 1. 0. 1. 2.]
 [2. 1. 1. 0. 1.]
 [1. 1. 2. 1. 0.]]
'''
distance_matrix_2 = get_distance_matrix(G2)
"""
[[0. 1. 1. 2. 2. 2. 3. 2. 3. 3.]
 [1. 0. 1. 1. 1. 2. 2. 2. 2. 3.]
 [1. 1. 0. 2. 1. 1. 2. 1. 3. 2.]
 [2. 1. 2. 0. 1. 2. 1. 2. 1. 2.]
 [2. 1. 1. 1. 0. 1. 1. 1. 2. 2.]
 [2. 2. 1. 2. 1. 0. 2. 1. 3. 2.]
 [3. 2. 2. 1. 1. 2. 0. 1. 1. 1.]
 [2. 2. 1. 2. 1. 1. 1. 0. 2. 1.]
 [3. 2. 3. 1. 2. 3. 1. 2. 0. 1.]
 [3. 3. 2. 2. 2. 2. 1. 1. 1. 0.]]
"""


def get_subgraph_distance_matrix(distance_matrix, point_ind_l):
    node_num = len(point_ind_l)
    sub_distance_matrix = np.zeros((node_num, node_num))
    for pair in itertools.combinations(list(range(node_num)), 2):
        p1_ind = point_ind_l[pair[0]]
        p2_ind = point_ind_l[pair[1]]
        distance = distance_matrix[p1_ind, p2_ind]
        sub_distance_matrix[pair] = distance
        sub_distance_matrix[pair[::-1]] = distance

    return sub_distance_matrix


while True:
    point2_ind_list = []
    for i in range(len(A1)):
        j = torch.argmax(X[i]).item()

        # 1. class checking
        # if class(j) != class(i)
        # x[i][j] = 0
        # j = torch.argmax(X[i]).item()
        # until class(j) = class(i)

        # 2. distance checking
        point2_ind_list.append(j)

    # 比较两个距离矩阵
    subgraph_distance_matrix = get_subgraph_distance_matrix(distance_matrix_2, point2_ind_list)
    result = np.absolute(np.array(distance_matrix_1) - np.array(subgraph_distance_matrix))
    # print(result)
    # print(result.sum(axis=1))
    # print(result.sum(axis=0))

    ok = True
    for point1_ind, diff_val in enumerate(result.sum(axis=1)):
        print(f"diff_val: {diff_val}")
        if diff_val > 10:
            X[point1_ind][point2_ind_list[point1_ind]] = 0
            ok = False
            break
    if ok:
        print('结束')
        break

img_name = os.path.splitext(os.path.basename(small_img_path))[0]


# MAD法: media absolute deviation
def MAD(dataset, n):
    median = np.median(dataset)  # 中位数
    deviations = abs(dataset - median)
    mad = np.median(deviations)

    remove_idx = np.where(abs(dataset - median) > n * mad)

    return remove_idx


sub_points_list = points2[point2_ind_list]
# 删除离群点
dis_matrix1 = distance.cdist(sub_points_list, sub_points_list, 'euclidean')
dis_l = dis_matrix1.sum(axis=1)
print(f'dis_l: {dis_l}')
remove_idx = MAD(dis_l, 3)
print(f'remove_idx: {remove_idx}')
sub_points_list = np.delete(sub_points_list, remove_idx, axis=0)

min_point = sub_points_list.min(axis=0).astype(np.int32)
max_point = sub_points_list.max(axis=0).astype(np.int32)

cv2.rectangle(big, tuple(min_point), tuple(max_point), (255, 0, 255), 5)
cv2.imwrite(f"{img_name}_result.jpg", big)  #

for i in range(len(A1)):
    j = torch.argmax(X[i]).item()

    con = ConnectionPatch(
        xyA=points1[i],
        xyB=points2[j],
        coordsA="data",
        coordsB="data",
        axesA=ax1,
        axesB=ax2,
        color="red"
    )
    plt.gca().add_artist(con)
plt.savefig(f'{img_name}_match_rrwm_v2.jpg')
plt.show()
