#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class YoloDetector {
public:
    /**
     * @brief 构造函数，初始化YOLO检测器
     * 
     * @param engine_path TensorRT引擎文件路径
     * @param conf_thresh 置信度阈值
     * @param nms_thresh NMS阈值
     */
    YoloDetector(const std::string& engine_path, float conf_thresh = 0.5, float nms_thresh = 0.45);
    
    /**
     * @brief 析构函数，释放资源
     */
    ~YoloDetector();
    
    /**
     * @brief 执行目标检测
     * 
     * @param image 输入图像(CV_8UC3格式)
     * @return 检测结果，每个向量包含[x, y, width, height, class_id, confidence]
     */
    std::vector<std::vector<float>> detect(const cv::Mat& image);
    
    // 禁用复制构造函数和赋值运算符
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

private:
    // 内部实现类 (PIMPL模式)
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // YOLO_DETECTOR_H