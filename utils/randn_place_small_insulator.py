import math
import os
import random
import json
import datetime

import cv2
import numpy as np

# train val
MODE = "val"

JPG_PATH = "../data/enhance/" + MODE + "/insulator"
OUT_PATH = "../data/enhance/" + MODE + "/enhanced"
LABELS_PATH = "../data/enhance/" + MODE + "/labels"
# JPG_PATH = "../data/test/ul/img"
# OUT_PATH = "../data/test/ul/ul_img"
# LABELS_PATH = "../data/test/ul/labels"

PNG_PATH = "../data/kou/cut"


def visualize(total, cur, imageFile):
    # [================>              ]  50
    print("\r" + imageFile + "\t\t" + "[", end='')
    progress = int((cur + 1) / total * 50)
    c_progress = '='
    c_remain = '-'
    ctr = '>'
    print(c_progress * progress + ctr + c_remain * (50 - progress), end='')
    print("]" + str(progress * 2) + "%", end='')
    if cur + 1 == total:
        print()


# 随机挑选20张绝缘子png，存入png_img_list中
def rand_select_png(png_path, n=20, resize=False):
    img_file_list = os.listdir(png_path)
    n = min(len(img_file_list), n)
    rand_idx = [random.randint(0, len(img_file_list) - 1) for i in range(n)]

    png_img_list = []
    # print("----- begin rand select png -----")
    for i, idx in enumerate(rand_idx):
        filename = img_file_list[idx]
        # print(filename)
        img = cv2.imread(os.path.join(png_path, filename), cv2.IMREAD_UNCHANGED)  # hwc
        if resize:
            # 图片尺寸缩小到 32 * 32 以内
            img = transform(img)
        png_img_list.append(img)
    # print("----- end rand select png -----")

    return png_img_list


# 绝缘子png缩小，水平、垂直翻转
# 32 64 128
# area < 32 * 32
def transform(img, size_min=32):
    sizes = 6 * [32] + 2 * [64] + [128]
    size_min = sizes[random.randint(0, len(sizes) - 1)]
    h = img.shape[0]
    w = img.shape[1]
    scale = h / w
    h_resize = int(math.sqrt((size_min ** 2) * scale))
    w_resize = int(math.sqrt((size_min ** 2) / scale))
    img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_AREA)
    return img


# 随机生成30个放置绝缘子图片的坐标（对应待叠加图的左上角坐标）
# [ (x0, y0), (x1, y1), ... ]
def generate_coordinate(img, n=20):
    h = img.shape[0]
    w = img.shape[1]
    coordinates = [(
        random.randint(0, w - 1), random.randint(0, h - 1)
    ) for i in range(n)]
    return coordinates


# 标记占位图 place_map (h, w)，返回标记是否成功
#   查找占位图，如果待放置图像超出原图大小 或者 待放置绝缘子存在与放置绝缘子存在交集的情况，continue
def mark_place_map(place_map: np.ndarray,
                   xmin, ymin, xmax, ymax) -> bool:
    xmin, ymin, xmax, ymax = float2int(xmin, ymin, xmax, ymax)

    h = place_map.shape[0]
    w = place_map.shape[1]
    if xmax > w or ymax > h:
        return False

    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            if place_map[i][j] == 1:
                return False

    # if np.any(place_map[ymin:ymax][xmin:xmax] == 1):
    #     return False

    place_map[ymin:ymax, xmin:xmax] = 1
    return True


def float2int(*floats):
    res = []
    for f in floats:
        res.append(int(f))
    return res


# 将png图叠加到jpg图指定位置
# h,w,c
# bgra
def add_png_in_jpg(jpg, png, xmin, ymin, xmax, ymax):
    # xmax和ymax本质上是numpy的长度，所以不可能取到
    # 已经在mark阶段过滤了超出图片的坐标，但是还是在此进行校验，增加系统健壮性
    h = jpg.shape[0]
    w = jpg.shape[1]
    ymax = min(h, ymax)
    xmax = min(w, xmax)
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            # 不透明
            if png[i - ymin, j - xmin, :][3] == 255:
                jpg[i, j, :] = [png[i - ymin, j - xmin, 0], png[i - ymin, j - xmin, 1], png[i - ymin, j - xmin, 2]]
    return jpg


# 将yolo坐标转换为coco坐标
def yolo2coco(W, H, x, y, w, h):
    x, y, w, h = float(x), float(y), float(w), float(h)  # 将字符串类型转为可计算的int和float类型

    x = round(x * W) + 0.0  # 1119
    y = round(y * H) + 0.0  # 2179
    w = (w * W) // 2 * 2 + 0.0  # 68
    h = (h * H) // 2 * 2 + 0.0  # 14
    xmin = abs((x - w / 2))  # 1085
    ymin = abs((y - h / 2))  # 2172
    xmax = abs((x + w / 2))  #
    ymax = abs((y + h / 2))  #

    return xmin, ymin, xmax, ymax


# 将coco坐标转换为yolo坐标
def coco2yolo(W, H, xmin, ymin, xmax, ymax):
    # x_ = (x1 + x2) / 2w
    # y_ = (y1 + y2) / 2h
    # w_ = (x2 - x1) / w
    # h_ = (y2 - y1) / h
    x = (xmin + xmax) / (2 * W)
    y = (ymin + ymax) / (2 * H)
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H

    return x, y, w, h


# 拿到图片和txt
# 拿到该图片标注绝缘子的坐标，在占位图上标记
# 遍历放置坐标，创建一个“占位图” place_map (h, w)，标记已被占用的位置。
#   随机从list挑选一张png进行叠加，叠加图片时，将已占位的位置设为1
#   将增强的绝缘子坐标信息追加到该图片对应的txt中
def main():
    jpg_file_list = os.listdir(JPG_PATH)
    for cur, filename in enumerate(jpg_file_list):
        # print("============ begin ============")
        # print(filename)
        visualize(len(jpg_file_list), cur, filename)
        img = cv2.imread(os.path.join(JPG_PATH, filename), cv2.IMREAD_UNCHANGED)  # hwc
        H = img.shape[0]
        W = img.shape[1]
        place_map = np.zeros((H, W), dtype=np.int32)

        txt_file = filename.split(".")[0] + '.txt'  # 获取该图片获取的txt文件
        with open(os.path.join(LABELS_PATH, txt_file), 'r') as fr:
            lines = fr.readlines()  # 读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
        for j, line in enumerate(lines):
            class_id, x, y, w, h = line.strip().split(' ')  # 获取每一个标注框的详细信息
            xmin, ymin, xmax, ymax = yolo2coco(W, H, x, y, w, h)
            # 标记占位图
            mark_place_map(place_map, xmin, ymin, xmax, ymax)

        # 随机选择待增强绝缘子
        png_img_list = rand_select_png(PNG_PATH, resize=True)
        # 生成随机左上角坐标
        coordinates = generate_coordinate(img)
        for coordinate in coordinates:
            # 随机选择一个png
            png_img = png_img_list[random.randint(0, len(png_img_list) - 1)]
            png_h = png_img.shape[0]
            png_w = png_img.shape[1]

            xmin = coordinate[0]
            ymin = coordinate[1]
            xmax = xmin + png_w
            ymax = ymin + png_h
            # 没有标记成功，使用下一个坐标
            if not mark_place_map(place_map, xmin, ymin, xmax, ymax):
                continue
            # 标记成功
            # 叠加
            img = add_png_in_jpg(img, png_img, xmin, ymin, xmax, ymax)
            # 追加保存yolo坐标信息
            x, y, w, h = coco2yolo(W, H, xmin, ymin, xmax, ymax)
            with open(os.path.join(LABELS_PATH, txt_file), 'a+') as fr:
                fr.write("0" + " " + str(format(x, '.6f')) + " " + str(format(y, '.6f')) + " " + str(format(w, '.6f')) + " " + str(format(h, '.6f')) + "\n")

        # 保存增强图片
        cv2.imwrite(os.path.join(OUT_PATH, filename), img)

        # print("============ end ============")


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print("执行时间：" + str((end - start).seconds) + "s")


# =======================================================================================================
# 拿到图片和anno
#   在images中找到对应图片的image_id
#   在annos中找到对应图片id的最后一个anno
#   将坐标信息插入到上述anno后
# 拿到该图片标注绝缘子的坐标，在占位图上标记
# 遍历放置坐标，创建一个“占位图” place_map (h, w)，标记已被占用的位置。
#   随机从list挑选一张png进行叠加，叠加图片时，将已占位的位置设为1
#   将增强的绝缘子坐标信息存入该图片对应的anno中
# jpg_file_list = os.listdir(JPG_PATH)
# with open(ANNO_PATH) as f:
#     # images = anno['images']: list
#     # images[0]['id']    images[0]['file_name']
#     # annos = anno[annotations]: list
#     # annos[0]['id']    annos[0]['image_id']    category_id  iscrowd  area  bbox  segmentation
#     anno = json.load(f)

