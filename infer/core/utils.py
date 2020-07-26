import tensorflow as tf
import numpy as np
import os
import random
import colorsys
import base64
import cv2

# 数据集识别80类别
CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
            29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
            48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'sofa', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
            62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# 图像预处理 大小统一
def image_preporcess(image, target_size):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.
    
    return image_paded[np.newaxis, ...].astype(np.float32)

# 图片np转base64，在cv中加色还原灰图
def image_to_base64(image_np):
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image = cv2.imencode('.jpg', image)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return 'data:image/jpeg;base64,{}'.format(image_code)

# 随机生成类别色
def random_colors(N):
    hsv_tuples = [(1.0 * x / N, 1., 1.) for x in range(N)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    return colors

# 绘制检测框
def draw_bbox(image, results, show_label=True):
    colors = random_colors(81)  # 生成80类颜色组
    recognizer = {}             # 识别物统计

    image_h, image_w = image.shape[:2]
    print("识别目标：{0} , 图片高宽：{1} x {2}".format(len(results), image_h, image_w))

    for i, bbox in enumerate(results):
        class_id = int(bbox[5])                             # 类别下标
        box_label = CLASSES[class_id]                       # 类别标签名称
        box_color = colors[class_id]                        # 类别所属颜色
        classes_score = bbox[4]                             # 类别识别分数
        y1, x1, y2, x2 = np.array(bbox[:4], dtype=np.int32) # 边框四点坐标

        # 是否识别并统计
        if box_label not in recognizer: recognizer[box_label] = []
        
        # 同类型追加
        label = recognizer[box_label]
        item = {}
        item['classes_score'] = str(round(classes_score,2))
        item['box_color'] = box_color
        item['box'] = np.array([y1, x1, y2, x2]).tolist()
        label.append(item)

        if show_label:
            box_size = 2        # 边框大小
            font_scale = 0.4    # 字体比例大小
            caption = '{} {:.2f}'.format(box_label, classes_score) if classes_score else box_label
            image = cv2.rectangle(image, (y1, x1), (y2, x2), box_color, box_size)
            # 填充文字区
            text_size = cv2.getTextSize(caption, 0, font_scale, thickness=box_size)[0]
            image = cv2.rectangle(image, (y1, x1), (y1 + text_size[0], x1 + text_size[1] + 8), box_color, -1)
            image = cv2.putText(
                image,
                caption,
                (y1, x1 + text_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (50, 50, 50),
                box_size//2,
                lineType=cv2.LINE_AA
            )

    # print(recognizer)    
    return image, recognizer

# 识别物体保存小图
def sliceImage(image_orig, recognizer, out_path):
    # 打开文件统计后遍历物体结果数据
    totalFile = open(out_path + "/totalCount.txt","w")
    for i, label in enumerate(recognizer):
        # 文件统计写入
        labelTotal = "{0}：{1} \n".format(label, len(recognizer[label]))
        totalFile.write(labelTotal)
        # 输出类别目录，不存在需要创建
        label_path = os.path.join(out_path, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        # 开始对类别物体进行切割
        for i, boxs in enumerate(recognizer[label]):
            y1, x1, y2, x2 = boxs['box']
            cv2.imwrite("{0}/{1}-{2}-{3}.jpg".format(label_path, label, i, boxs['classes_score']), image_orig[x1:x2, y1:y2])
    # 关闭文件统计        
    totalFile.close()
