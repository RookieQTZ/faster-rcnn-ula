import os

import cv2
import sys

PATH = "../data/kou"
OUT_PATH = "../data/kou/cut"

img_file_list = os.listdir(PATH)
for filename in img_file_list:
    if not filename.endswith(".png"):
        continue
    print(filename)
    base = filename.split(".")[0]
    img = cv2.imread(os.path.join(PATH, filename), cv2.IMREAD_UNCHANGED)  # hwc

    # 找到xmin xmax ymin ymax
    h = img.shape[0]
    w = img.shape[1]

    xmin = ymin = sys.maxsize
    xmax = ymax = 0

    for i in range(0, h):
        for j in range(0, w):
            c = img[i, j, :]
            if c[3] == 255:
                xmin = min(xmin, j)
                xmax = max(xmax, j)
                ymin = min(ymin, i)
                ymax = max(ymax, i)

    cropped = img[ymin:ymax, xmin:xmax]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(OUT_PATH, filename), cropped)
