import random
import cv2
import numpy as np
import onnxruntime as ort
import time

class Yolov9(object):
    def __init__(self):
        self.model_pb_path = "yolov9-t-converted.onnx"  # 模型加载
        self.so = ort.SessionOptions()
        self.session = ort.InferenceSession(self.model_pb_path, self.so)
        self.conf_thres = 0.75
        self.model_shape = (640, 640)
        # 标签字典
        self.dic_labels = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }

    # 解析向量
    def prcess_net_infos(self, prediction, target_cls=17):
        # bs = prediction.shape[0]          # batch size
        # nc = prediction.shape[1] - 4      # number of classes
        mi = prediction.shape[1]            # mask start index
        confidence = prediction[:, 4:mi]    # confidence=(80, 8400) ; 表示有8400个框，每个框给80个类别打分(百分数)
        xc = confidence.max(axis=1) > self.conf_thres    # 每个框，取预测类别最大的概率值 与 conf_thres进行比较，小于conf_thres认为，当前框没有预测到有效类别，直接过滤
        filter_tag = xc[0]

        x_infos = prediction[0]       # (1, 84, 8400) --> (84, 8400)
        x = x_infos.T[filter_tag]     # confidence   x (84, 8400) --> (n, 84); 安照行标记过滤以后,把检测到物体的n个框留下

        if not x.shape[0]:
            return None

        cls = x[:, 4:84]                # 取第5列到第84列，形状为 (n, 80)
        j = np.argmax(cls, axis=1)      # 取出每组预测值(80个)最大的类别编号(0-79)
        class_tag = j == target_cls     # 安装指定类别号在过滤一次
        final_result = x[class_tag]

        if not x.shape[0]:
            return None

        # box = final_result[:, :4]       # 取前4列 box=(x,y,h,w)，形状为 (row_n, 4)
        cls = final_result[:, 4:84]
        conf = np.max(cls, axis=1, keepdims=True)

        if len(conf) > 0 and len(conf[0]) > 0:
            label = f'{self.dic_labels[target_cls]} {conf[0][0]:.2f}'
            return label

        return None

    # 图片预处理
    def transform(self, im0):
        im = cv2.resize(im0, self.model_shape, interpolation=cv2.INTER_LINEAR)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype(np.float32)
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im


if __name__ == "__main__":
    yolov9 = Yolov9()
    cap = cv2.VideoCapture(0)
    while True:
        success, img0 = cap.read()
        if success:
            # _, img0 = getdata()
            t1 = time.time()
            img = yolov9.transform(img0)
            pred = yolov9.session.run(None, {'images': img})[0]
            leable = yolov9.prcess_net_infos(pred, target_cls=0)
            t2 = time.time()

            str_FPS = "FPS: %.2f" % (1. / (t2 - t1))
            if leable is not None:
                cv2.putText(img0, leable, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
            else:
                cv2.putText(img0, str_FPS, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow("video", img0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


    cap.release()