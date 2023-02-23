import os.path

import numpy as np
import cv2


UL_IMG_PATH = "../data/ul_img"
OUT_PATH = "../data/ul_img"


def bin_img(path, filename):
    # 读取图片。
    img = cv2.imread(os.path.join(path, filename))
    # 转成灰度图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binimg = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # # 显示图像
    # cv2.imshow("title", img)
    # # 进程不结束，一直保持显示状态
    # cv2.waitKey(0)
    # # 销毁所有窗口
    # cv2.destroyAllWindows()
    return binimg


def binimg2png(binimg, base, out_path):
    '''
    :param binimg: 二值图像
    :param base: 文件名（不含后缀）
    :param out_path: 输出路径
    :return:
    '''
    h = binimg.shape[0]
    w = binimg.shape[1]
    np_img = np.empty((4, h, w), dtype=int)  # (c, h, w)

    for i in range(0, h):
        for j in range(0, w):
            if binimg[i, j] == 255:
                # 不透明
                # 完全透明的像素应该将alpha设置为0，完全不透明的像素应该将alpha设置为255/65535。
                np_img[:, i, j] = [binimg[i, j], binimg[i, j], binimg[i, j], 255]  # rgba
            else:
                np_img[:, i, j] = [255, 255, 255, 0]

    # rgba -> bgra
    r = np_img[0, :, :]
    g = np_img[1, :, :]
    b = np_img[2, :, :]
    a = np_img[3, :, :]
    np_img = np.stack((b, g, r, a), axis=2)  # h w c

    # 格式转换操作，确保可以从numpy数组转换为img
    # imwrite函数只支持保存8位（或16位无符号）
    np_img = np_img.astype(np.uint8)
    cv2.imwrite(os.path.join(out_path, "kou_" + base + ".png"), np_img)


def ulimg2png(in_path, out_path):
    img_file_list = os.listdir(in_path)
    for filename in img_file_list:
        print(filename)
        base = filename.split(".")[0]
        binimg = bin_img(in_path, filename)
        binimg2png(binimg, base, out_path)


ulimg2png(UL_IMG_PATH, OUT_PATH)
