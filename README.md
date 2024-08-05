本项目基于yolov9项目，在yolov9的基础上做了如下尝试

0，yolov9 项目地址
(https://github.com/WongKinYiu/yolov9)

1，将xxx.pt格式模型文件转换成  xxx.onnx 格式文件 (yolov9 项目当中提供 export.py)

  python export.py --device cpu --weights './yolov9-m.pt' --include onnx
  

2, 在yolov9项目当中，下载好模型，并测试(这里测试的是 yolov9-m.pt)

  python detect_dual.py --source './data/images/2.jpeg' --img 640 --device cpu --weights './yolov9-m.pt' --name yolov9_c_640_detect
  

3，xxx.py脚本说明

  1) success_onnx_nms.py      # onnx模型在pytorch，GPU上运行
  2) success_onnx_nms_cpu.py    # onnx模型在numpy CPU上运行
  3) xr_yolov9_detection.py     # 可以实时接收摄像头设备的图像，进行物体检测


4, success_onnx_nms_cpu.py  、 xr_yolov9_detection.py 的运行环境

Python 3.7.16

Name: numpy
Version: 1.21.6


