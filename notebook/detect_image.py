import os, sys
import argparse
import tensorflow as tf
from moxing.framework import file
import cv2
import time

# 执行参数 python notebook/detect_image.py --image notebook/test.jpg --min_score 0.3 --show_box_label true --input_size 608 --version V0015
# 外部参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='notebook/test.jpg', help='data jpg file.')
parser.add_argument('--min_score', type=float, default=0.3, help='show minimum score.')
parser.add_argument('--show_box_label', type=bool, default=True, help='show identification border labels.')
parser.add_argument('--input_size', type=int, default=608, help='training uniform picture size. [608 512 416 320]')
parser.add_argument('--version', type=str, default='V0015', help='model version')
ARGS = parser.parse_args()

# 执行所在路径， V0xxx 表示模型版本号
source_path = os.path.join(os.getcwd(), "model/" + ARGS.version + "/model")
sys.path.append(source_path)

from core.utils import image_preporcess, draw_bbox, image_to_base64, sliceImage
from core.yolov4 import model_load, detect

# 统一输入图片大小
input_size = ARGS.input_size

created_at = str(round(time.time() * 1000))

# obs桶路径
obs_path = "obs://puddings/ma-yolov4/notebook/out/image/" + created_at

# 输出目录
out_path = "notebook/out/image/" + created_at

# 输出目录存在需要删除里边的内容
if os.path.exists(out_path):
    file.remove(out_path, recursive=True)
os.makedirs(out_path)

if __name__ == "__main__":
    # 载入模型
    model = model_load(os.path.join(source_path, 'yolov4.weights'), input_size)

    # 读取图片
    image = cv2.imread(ARGS.image)
    # 原图用于分割小图
    image_orig = image.copy() 

    prev_time = time.time()
    
    # 模型识别结果
    image_data = image_preporcess(image.copy(), [input_size, input_size])
    predict = model.predict(image_data)

    # 结果绘制到图
    results = detect(predict, image.shape[:2], input_size, ARGS.min_score)
    image, recognizer = draw_bbox(image, results,  ARGS.show_box_label)

    # 绘制时间
    curr_time = time.time()
    exec_time = curr_time - prev_time
    print("识别耗时: %.2f ms" %(1000*exec_time))

    # print("识别结果：", recognizer)

    # 输入图片np uint8 尺寸/2
    # x, y = image.shape[0:2]
    # image = cv2.resize(image, (int(y / 2), int(x / 2)))

    # base图片编码
    # itb64 = image_to_base64(image)
    # print(itb64)
    
    # 绘制识别统计
    totalStr = ""
    for k in recognizer.keys():
        totalStr += '%s: %d    ' % (k, len(recognizer[k]))
    cv2.putText(image, totalStr, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

    # 绘制保存
    cv2.imwrite(out_path + "/output_result.jpg", image)
    cv2.imwrite(out_path + "/output_orig.jpg", image_orig)

    # 切割识别到的物体
    sliceImage(image_orig, recognizer, out_path)

    # 复制保存到桶
    print("输出目录：" + out_path)
    file.copy_parallel(out_path, obs_path)
    
    # 显示窗口
    # cv2.namedWindow('image_result', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('image_result', image)
    # 退出窗口
    # cv2.waitKey(0)
    # 任务完成后释放内容
    # cv2.destroyAllWindows()
    