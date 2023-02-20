import os
import json
import cv2
import random
import time
from PIL import Image

# train2017 val2017
MODE = "val2017"

# coco_format_save_path = 'F:/workspace_pycharm/SODnet/faster-rcnn/data/coco2017/annotations'  # 要生成的标准coco格式标签所在文件夹
# yolo_format_classes_path = 'F:/workspace_pycharm/SODnet/faster-rcnn/data/coco2017/classes.txt'  # 类别文件，一行一个类
# yolo_format_annotation_path = 'F:/workspace_pycharm/SODnet/faster-rcnn/data/SFID/labels/' + MODE  # yolo格式标签所在文件夹
# img_pathDir = 'F:/workspace_pycharm/SODnet/faster-rcnn/data/coco2017/' + MODE  # 图片所在文件夹
coco_format_save_path = 'F:/workspace_pycharm/SODnet/faster-rcnn/data/test/annotations'  # 要生成的标准coco格式标签所在文件夹
yolo_format_classes_path = 'F:/workspace_pycharm/SODnet/faster-rcnn/data/test/classes.txt'  # 类别文件，一行一个类
yolo_format_annotation_path = 'F:/workspace_pycharm/SODnet/faster-rcnn/data/test/labels/' + MODE  # yolo格式标签所在文件夹
img_pathDir = 'F:/workspace_pycharm/SODnet/faster-rcnn/data/test/' + MODE  # 图片所在文件夹


def visualize(total, cur, imageFile):
    # [================>              ]  50
    print("\r" + imageFile + "\t\t" + "[", end='')
    progress = int(cur / total * 50)
    c_progress = '='
    c_remain = '-'
    ctr = '>'
    print(c_progress * progress + ctr + c_remain * (50 - progress), end='')
    print("]" + str(progress * 2) + "%", end='')



with open(yolo_format_classes_path, 'r') as fr:  # 打开并读取类别文件
    lines1 = fr.readlines()
# print(lines1)
categories = []  # 存储类别的列表
for j, label in enumerate(lines1):
    label = label.strip()
    categories.append({'id': j + 1, 'name': label, 'supercategory': 'None'})  # 将类别信息添加到categories中
# print(categories)

write_json_context = dict()  # 写入.json文件的大字典
write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2023, 'contributor': 'kyten',
                              'date_created': '2023-02-17'}
write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
write_json_context['categories'] = categories
write_json_context['images'] = []
write_json_context['annotations'] = []

# 接下来的代码主要添加'images'和'annotations'的key值
imageFileList = os.listdir(img_pathDir)  # 遍历该文件夹下的所有文件，并将所有文件名添加到列表中
for i, imageFile in enumerate(imageFileList):
    if not imageFile.endswith(".jpg"):
        continue
    # jpg_count = sum([filename.endswith(".jpg") for filename in imageFileList])
    visualize(len(imageFileList), i + 1, imageFile)  # 显示进度
    imagePath = os.path.join(img_pathDir, imageFile)  # 获取图片的绝对路径
    image = Image.open(imagePath)  # 读取图片，然后获取图片的宽和高
    W, H = image.size

    img_context = {}  # 使用一个字典存储该图片信息
    # img_name=os.path.basename(imagePath)                                       #返回path最后的文件名。如果path以/或\结尾，那么就会返回空值
    img_context['file_name'] = imageFile
    img_context['height'] = H
    img_context['width'] = W
    img_context['date_captured'] = '2023-02-17'
    img_context['id'] = i  # 该图片的id
    img_context['license'] = 1
    img_context['coco_url'] = ''
    img_context['flickr_url'] = ''
    write_json_context['images'].append(img_context)  # 将该图片信息添加到'image'列表中

    txtFile = imageFile.split(".")[0] + '.txt'  # 获取该图片获取的txt文件
    with open(os.path.join(yolo_format_annotation_path, txtFile), 'r') as fr:
        lines = fr.readlines()  # 读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
    for j, line in enumerate(lines):
        bbox_dict = {}  # 将每一个bounding box信息存储在该字典中
        # line = line.strip().split()
        # print(line.strip().split(' '))

        class_id, x, y, w, h = line.strip().split(' ')  # 获取每一个标注框的详细信息
        class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)  # 将字符串类型转为可计算的int和float类型

        x = round(x * W) + 0.0  # 641
        y = round(y * H) + 0.0  # 516
        w = (w * W) // 2 * 2 + 0.0  # 744
        h = (h * H) // 2 * 2 + 0.0  # 218
        xmin = abs((x - w / 2))  # 269
        ymin = abs((y - h / 2))  # 407
        xmax = abs((x + w / 2))  # 1013
        ymax = abs((y + h / 2))  # 625

        # xmin = abs(round((x - w / 2) * W)) + 0.0  # 坐标转换 1.01 -> 1.0    1.91 -> 2.0
        # ymin = abs(round((y - h / 2) * H)) + 0.0
        # xmax = abs(round((x + w / 2) * W)) + 0.0
        # ymax = abs(round((y + h / 2) * H)) + 0.0
        # w = round(w * W) + 0.0
        # h = round(h * H) + 0.0

        bbox_dict['id'] = i * 10000 + j  # bounding box的坐标信息
        bbox_dict['image_id'] = i
        # bbox_dict['category_id'] = class_id + 1  # 注意目标类别要加一
        bbox_dict['category_id'] = 1  # 注意目标类别要加一
        bbox_dict['iscrowd'] = 0
        # height, width = abs(ymax - ymin), abs(xmax - xmin)
        bbox_dict['area'] = w * h
        bbox_dict['bbox'] = [xmin, ymin, w, h]
        bbox_dict['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
        write_json_context['annotations'].append(bbox_dict)  # 将每一个由字典存储的bounding box信息添加到'annotations'列表中

name = os.path.join(coco_format_save_path, "instances_" + MODE + '.json')
with open(name, 'w') as fw:  # 将字典信息写入.json文件中
    json.dump(write_json_context, fw, indent=2)
print("\ncongratulations: yolo2coco successful!!")
