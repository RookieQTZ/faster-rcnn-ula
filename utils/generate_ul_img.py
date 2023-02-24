import math
import os
import datetime
import cv2
import random
import numpy as np

import randn_place_small_insulator as utils

N = 5

# 增强图个数分别为：
#   3498
#   888
#####################
MODE = "val2017"  # train2017 val2017
NUM_UL_IMG = 800   # 3000 800
#####################

IMG_PATH = "../data/coco2017/" + MODE
LABELS_PATH = "../data/SFID/labels/" + MODE
OUT_PATH = "../data/generate_ul_img/" + MODE
UL_PATH = "../data/ul_img"
# IMG_PATH = "../data/test/ul/img"
# LABELS_PATH = "../data/test/ul/labels"
# OUT_PATH = "../data/test/ul/ul_img"
# UL_PATH = "../data/ul_img"


# 随机挑选20张绝缘子png，存入png_img_list中
def rand_select_png(png_path, n=20, resize=False):
    img_file_list = os.listdir(png_path)
    n = min(len(img_file_list), n)
    rand_idx = [random.randint(0, len(img_file_list) - 1) for i in range(n)]

    png_img_list = []
    # print("----- begin rand select png -----")
    for i, idx in enumerate(rand_idx):
        filename = img_file_list[idx]
        if filename.startswith("kou"):
            # print(filename)
            img = cv2.imread(os.path.join(png_path, filename), cv2.IMREAD_UNCHANGED)  # hwc
            if resize:
                # 图片尺寸缩小到 32 * 32 以内
                img = transform(img)
            png_img_list.append(img)
    # print("----- end rand select png -----")

    return png_img_list


# 将图像缩放到指定尺度以下
# 32 64 128
def transform(img, size_min=16):
    sizes = [16, 32, 64, 128]
    ran = 0
    if size_min <= 32:
        ran = 0
    elif size_min <= 64:
        ran = 1
    elif size_min <= 128:
        ran = 2
    else:
        ran = 3

    # size_min = sizes[random.randint(0, ran)]
    # h = img.shape[0]
    # w = img.shape[1]
    # scale = h / w
    # h_resize = int(math.sqrt((size_min ** 2) * scale))
    # w_resize = int(math.sqrt((size_min ** 2) / scale))
    # resize_img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_AREA)
    # return resize_img

    size_min = sizes[random.randint(0, ran)]
    h = img.shape[0]
    w = img.shape[1]
    long_side = max(h, w)
    scale = size_min / long_side
    resize_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resize_img


# 在尺度内，随机生成一个左上角坐标
# (x0, y0)
def generate_coordinate(xmin, ymin, xmax, ymax):
    xmin, ymin, xmax, ymax = utils.float2int(xmin, ymin, xmax, ymax)
    return random.randint(xmin, xmax - 1), random.randint(ymin, ymax - 1)


# 将png图叠加到jpg图指定位置
# h,w,c
# bgra
def add_png_in_jpg(jpg, png, xmin, ymin, xmax, ymax):
    # xmax和ymax本质上是numpy的长度，所以不可能取到
    # hwc
    h = jpg.shape[0]
    w = jpg.shape[1]
    ymax = min(h, ymax)
    xmax = min(w, xmax)
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            # 白色并且不透明
            if png[i - ymin, j - xmin, :][3] == 255\
                    and png[i - ymin, j - xmin, :][0] == 255\
                    and png[i - ymin, j - xmin, :][1] == 255\
                    and png[i - ymin, j - xmin, :][2] == 255:
                jpg[i, j, :] = [255, 255, 255]
    return jpg


# 将ul图存入ul_list中
# 拿到图片（train：2400，val：800）和txt
# 获取图片中所有绝缘子的尺度信息
#   随机选择N个绝缘子模拟紫外放电
#   根据尺度信息，随机将ul图缩放到指定大小以下
#   创建一个与原图相同大小的arr，在尺度信息内，选择随机位置将ul图添加上去
def main():
    ul_list = rand_select_png(UL_PATH)
    img_file_list = os.listdir(IMG_PATH)

    for cur, filename in enumerate(img_file_list):
        utils.visualize(len(img_file_list), cur, filename)
        # 拿到图片
        img = cv2.imread(os.path.join(IMG_PATH, filename), cv2.IMREAD_UNCHANGED)  # hwc
        H = img.shape[0]
        W = img.shape[1]

        # 黑色背景
        ul_img = np.ndarray((H, W, 3))
        ul_img[:, :, :] = [0, 0, 0]

        # TRAIN 张以内
        if cur <= NUM_UL_IMG:
            txt_file = filename.split(".")[0] + '.txt'
            with open(os.path.join(LABELS_PATH, txt_file), 'r') as fr:
                lines = fr.readlines()
            n = min(N, len(lines))  # 生成n个ul图
            # 随机选择n张ul图
            lines = [lines[random.randint(0, len(lines) - 1)]
                     for i in range(n)]
            for j, line in enumerate(lines):
                # 随机取一张ul图
                ul = ul_list[random.randint(0, len(ul_list) - 1)]

                class_id, x, y, w, h = line.strip().split(' ')  # 获取每一个标注框的详细信息
                xmin, ymin, xmax, ymax = utils.yolo2coco(W, H, x, y, w, h)
                long_side = max(xmax - xmin, ymax - ymin)
                # 生成的ul面积要比当前绝缘子面积小
                ul = transform(ul, long_side)
                png_h = ul.shape[0]
                png_w = ul.shape[1]
                xmin, ymin = generate_coordinate(xmin, ymin, xmax, ymax)
                xmax = xmin + png_w
                ymax = ymin + png_h
                # 添加到紫外放电图上
                ul_img = add_png_in_jpg(ul_img, ul, xmin, ymin, xmax, ymax)

        # 保存ul_img
        # ul_1.jpg
        cv2.imwrite(os.path.join(OUT_PATH, "ul_" + filename), ul_img)


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print("执行时间：" + str((end - start).seconds) + "s")

# ul_list = rand_select_png(UL_PATH)
# ul = ul_list[0]
# cv2.imshow("origin", ul)
#
# resize_ul = transform(ul)
# cv2.imshow("resize", resize_ul)
# cv2.waitKey()
# cv2.destroyAllWindows()
