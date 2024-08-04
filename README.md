本项目基于yolov9项目，在yolov9的基础上做了如下尝试

0，yolov9 项目地址
(https://github.com/WongKinYiu/yolov9)

1，将xxx.pt格式模型文件转换成  xxx.onnx 格式文件 (yolov9 项目当中提供 export.py)
python export.py --device cpu --weights './yolov9-m.pt' --include onnx

2, 在yolov9项目当中，下载好模型，并测试(这里测试的是 yolov9-m.pt)
python detect_dual.py --source './data/images/2.jpeg' --img 640 --device cpu --weights './yolov9-m.pt' --name yolov9_c_640_detect

