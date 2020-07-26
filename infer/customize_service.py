from model_service.tfserving_model_service import TfServingBaseService
from core.utils import image_preporcess, draw_bbox, image_to_base64, sliceImage
from core.yolov4 import model_load, detect
from skimage import io

# 推理服务
class yolov4_service(TfServingBaseService):
    count = 1               # 预测次数
    model_object = None     # 模型实例
    input_size = 608        # 处理数据值

    def _preprocess(self, data):
        temp_data = {}
        
        # 遍历提交参数取值，image必传，配置默认值
        for k, v in data.items():
            if k == 'image':
                # 参数的默认值
                temp_data['min_score'] = float(data['min_score']) if 'min_score' in data else 0.2
                temp_data['show_image'] = int(data['show_image']) if 'show_image' in data else 1
                temp_data['show_box_label'] = int(data['show_box_label']) if 'show_box_label' in data else 1
                temp_data['input_size'] = int(data['input_size']) if 'input_size' in data else 608
                
                # file_name, file_content 图片字典数据
                for _, file_content in v.items():
                    image = io.imread(file_content)
                    temp_data[k] = image

        # 变更模型实例
        if(self.input_size != temp_data['input_size']):
            print('--变更模型实例--')
            self.input_size = temp_data['input_size']
            self.model_object = model_load('model/1/yolov4.weights', temp_data['input_size'])

        # 加载模型实例
        if(self.model_object == None):
            print('--加载模型实例--')
            self.model_object = model_load('model/1/yolov4.weights', temp_data['input_size'])

        return temp_data

    def _postprocess(self, data):
        outputs = {}

        # 输入参数
        image = data['image']
        input_size = data['input_size']

        # 模型识别结果
        image_data = image_preporcess(image.copy(), [input_size, input_size])
        predict = self.model_object.predict(image_data)

        # 结果绘制到图
        results = detect(predict, image.shape[:2], input_size, data['min_score'])
        image, recognizer = draw_bbox(image, results, data['show_box_label'])

        # 预测次数+1
        print('预测次数：', self.count)
        self.count += 1

        # 输出数据，show_image是否输出识别处理的base64图片
        outputs['recognizer_data'] = recognizer
        if data['show_image']:
            itb64 = image_to_base64(image)
            outputs['predicted_image'] = itb64

        return outputs

    def _inference(self, data):
        return data
