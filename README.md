# tensorrtx_yolov5-v7.0_ros
tensorrtx_yolov5-v7.0的ros2系统使用

# 使用方法


  
 - ros2编译启动
 - 使用tensorrtx＋yolov5转换.pt到.wts
 git clone https://github.com/ultralytics/yolov5.git
 git clone https://github.com/wang-xinyu/tensorrtx
 - 使用trt.cpp的main转换.wts到.engine


 # 注意
 1. 模型pt的训练精度 32 
 2. 模型转换时的参数：类别数量，图像尺寸，精度32
 3. 推理时使用同一个gpu