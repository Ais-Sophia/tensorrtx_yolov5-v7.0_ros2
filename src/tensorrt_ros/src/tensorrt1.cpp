#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>     // OpenCV头文件
#include <vector>   // 向量头文件

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include <iostream>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;
// 自定义消息类型 - 包含原始数据和检测结果
#include "/home/epoch/Desktop/1/R1_vision_v2/install/interfaces/include/interfaces/msg/tensorrt.hpp"
#include "interfaces/msg/tensorrt.hpp"

using namespace nvinfer1;
static Logger gLogger;
// 常量定义
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class YoloDetector {
public:
    /**
     * 构造函数
     * 
     * @param gpu_id 使用的GPU设备ID
     */
    explicit YoloDetector(int gpu_id = 0,
        const std::string& yolo_engine_path = "yolov5s.engine",
        bool enable_imshow = true
    ) {
        cudaSetDevice(gpu_id);
        init(yolo_engine_path);  // 默认引擎文件路径
        this->enable_imshow = enable_imshow;  // 是否启用imshow显示检测结果
    }

    /**
     * 从引擎文件初始化推理引擎
     * 
     * @param engine_path TensorRT引擎文件路径
     * @return true - 初始化成功, false - 初始化失败
     */
    bool init(const std::string& engine_path) {
        // 加载序列化的引擎文件
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "无法打开引擎文件: " << engine_path << std::endl;
            return false;
        }
        // 获取文件大小
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        // 读取文件内容
        char* serialized_engine = new char[size];
        file.read(serialized_engine, size);
        file.close();
        // 创建运行时和引擎
        runtime_ = createInferRuntime(gLogger);
        if (!runtime_) {
            std::cerr << "创建TensorRT运行时失败" << std::endl;
            delete[] serialized_engine;
            return false;
        }
        // 反序列化引擎
        engine_ = runtime_->deserializeCudaEngine(serialized_engine, size);
        delete[] serialized_engine;
        if (!engine_) {
            std::cerr << "反序列化CUDA引擎失败" << std::endl;
            return false;
        }
        // 创建执行上下文
        context_ = engine_->createExecutionContext();
        if (!context_) {
            std::cerr << "创建执行上下文失败" << std::endl;
            return false;
        }
        // 创建CUDA流
        CUDA_CHECK(cudaStreamCreate(&stream_));
        // 初始化CUDA预处理资源
        cuda_preprocess_init(kMaxInputImageSize);
        // 分配输入输出缓冲区
        prepareBuffers();
        
        return true;
    }

    /**
     * 执行单帧目标检测
     * 
     * @param frame 输入图像帧
     * @return 检测结果向量
     */
    std::vector<Detection> detect(const cv::Mat& frame) {
        // 预处理并执行推理
        std::cout << "开始预处理和推理..." << std::endl;
        preprocessAndInference(frame);
        std::cout << "推理完成，开始后处理..." << std::endl;
                    // 直接解析原始模型输出（跳过NMS后处理）
            // 1. 计算原始图像尺寸
    // int orig_w = frame.cols;
    // int orig_h = frame.rows;
    
    // // 2. 计算缩放比例
    // float scale_x = static_cast<float>(kInputW) / orig_w;
    // float scale_y = static_cast<float>(kInputH) / orig_h;
    // float scale = std::min(scale_x, scale_y);
    
    // // 3. 输出原始前20个值
    // std::cout << "原始输出前20个值: ";
    // for(int i = 0; i < 20; i++) {
    //     std::cout << cpu_output_buffer_[i] << " ";
    // }
    // std::cout << std::endl;
    
    // // 4. 解析输出格式
    // std::vector<Detection> detections;
    
    // // 假设输出格式为: [batch, num_detections, 85]
    // float* output = cpu_output_buffer_;
    // const int num_detections = kOutputSize / sizeof(float) / 85; // 85 = 4坐标 + 1置信度 + 80类别
    
    // // 5. 尝试不同解析策略
    // for (int i = 0; i < num_detections; i++) {
    //     float* det_ptr = output + i * 85;
        
    //     // 提取边界框 (cx, cy, w, h)
    //     float cx = det_ptr[0];
    //     float cy = det_ptr[1];
    //     float w = det_ptr[2];
    //     float h = det_ptr[3];
        
    //     // 目标置信度
    //     float obj_conf = det_ptr[4];
        
    //     // 获取类别置信度
    //     int class_id = -1;
    //     float max_class_conf = 0.0f;
    //     for (int c = 0; c < 80; c++) {
    //         float class_conf = det_ptr[5 + c];
    //         if (class_conf > max_class_conf) {
    //             max_class_conf = class_conf;
    //             class_id = c;
    //         }
    //     }
        
    //     // 跳过低置信度检测
    //     if (obj_conf < 0.001f) continue; // 临时降低阈值
        
    //     // 坐标转换 (从模型输入空间到原始图像空间)
    //     float x = (cx - (kInputW - orig_w * scale) / 2.0) / scale;
    //     float y = (cy - (kInputH - orig_h * scale) / 2.0) / scale;
    //     float width = w / scale;
    //     float height = h / scale;
        
    //     // 创建检测对象
    //     Detection det;
    //     det.bbox[0] = x;
    //     det.bbox[1] = y;
    //     det.bbox[2] = width;
    //     det.bbox[3] = height;
    //     det.conf = obj_conf;
    //     det.class_id = class_id;
    //     detections.push_back(det);
        
    //     // 打印调试信息
    //     std::cout << "有效检测框: "
    //               << "x=" << x << ", y=" << y << ", w=" << width << ", h=" << height
    //               << ", conf=" << obj_conf << ", class=" << class_id << std::endl;
    // }
    
    // return detections;
    // std::cout << "解析完成，返回原始检测结果（数量: " << detections.size() << "）" << std::endl;
        // 后处理
        std::vector<Detection> detections;
    std::vector<std::vector<Detection>> batch_detections;
    batch_detections.push_back(detections); // 将单帧检测结果包装成批处理格式
    batch_nms(batch_detections, cpu_output_buffer_, 1, kOutputSize, kConfThresh, kNmsThresh);
        
        return detections;
    }

    /**
     * 释放资源
     */
    ~YoloDetector() {
        // 释放GPU缓冲区
        if (gpu_input_buffer_) {
            CUDA_CHECK(cudaFree(gpu_input_buffer_));
        }
        if (gpu_output_buffer_) {
            CUDA_CHECK(cudaFree(gpu_output_buffer_));
        }
        
        // 释放CPU缓冲区
        if (cpu_output_buffer_) {
            delete[] cpu_output_buffer_;
        }
        
        // 销毁CUDA预处理资源
        cuda_preprocess_destroy();
        
        // 销毁CUDA流
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        
        // 销毁TensorRT资源
        if (context_) {
            delete context_;
        }
        if (engine_) {
            delete engine_;
        }
        if (runtime_) {
            delete runtime_;  
        }
    }

private:
    /**
     * 准备GPU/CPU缓冲区
     */
    void prepareBuffers() {
    // 1. 初始化变量
    input_tensor_name_ = nullptr;
    output_tensor_name_ = nullptr;
    std::cout << "引擎中IOTensor数量: " << engine_->getNbIOTensors() << std::endl;
    
    // 2. 遍历引擎中的所有IOTensor
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        // 获取张量名称和I/O模式
        const char* tensor_name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
        
        // 打印详细信息用于调试
        std::string io_type = (io_mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        std::cout << "Tensor " << i << " [" << io_type << "]: " << tensor_name << std::endl;
        
        // 复制输入/输出张量名称（使用strdup）
        if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
            if (input_tensor_name_) {
                free((void*)input_tensor_name_);
            }
            input_tensor_name_ = strdup(tensor_name);
            input_index_ = i;
            RCLCPP_INFO(rclcpp::get_logger("tensorrt"), "Input tensor: %s", input_tensor_name_);
        } 
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (output_tensor_name_) {
                free((void*)output_tensor_name_);
            }
            output_tensor_name_ = strdup(tensor_name);
            output_index_ = i;
            RCLCPP_INFO(rclcpp::get_logger("tensorrt"), "Output tensor: %s", output_tensor_name_);
        }
    }
    
    // 3. 检查是否找到输入和输出
    if (!input_tensor_name_ || !output_tensor_name_) {
        std::cerr << "错误：未找到输入或输出张量名称！" << std::endl;
        return;
    }
    
    // 4. 分配输入缓冲区（1x3xHxW）
    size_t input_size = 1 * 3 * kInputH * kInputW * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&gpu_input_buffer_, input_size));
    std::cout << "分配GPU输入缓冲区大小: " << input_size << " 字节" << std::endl;
    
    // 5. 分配输出缓冲区
    size_t output_size = 1 * kOutputSize * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&gpu_output_buffer_, output_size));
    std::cout << "分配GPU输出缓冲区大小: " << output_size << " 字节" << std::endl;
    
    // 6. 分配CPU输出缓冲区
    cpu_output_buffer_ = new float[1 * kOutputSize];
    std::cout << "分配CPU输出缓冲区大小: " << (1 * kOutputSize * sizeof(float)) << " 字节" << std::endl;
    
    // 7. 设置TensorRT上下文
    if (context_) {
        std::cout << "设置输入Tensor地址: " << input_tensor_name_ 
                  << " -> " << static_cast<void*>(gpu_input_buffer_) << std::endl;
        context_->setInputTensorAddress(input_tensor_name_, gpu_input_buffer_);
        
        std::cout << "设置输出Tensor地址: " << output_tensor_name_ 
                  << " -> " << static_cast<void*>(gpu_output_buffer_) << std::endl;
        context_->setTensorAddress(output_tensor_name_, gpu_output_buffer_);
    }
}

    /**
     * 图像预处理并执行推理
     * 
     * @param frame 输入图像帧
     */
    void preprocessAndInference(const cv::Mat& frame) {
        // 创建单元素的Mat向量
        imshow("原始图像", frame);
        std::vector<cv::Mat> frame_vector{frame};
        // std::cout << "图像预处理..." << std::endl;
        // CUDA预处理(将图像调整大小、归一化并复制到GPU)
        // std::cout << "Frame vector size: " << frame_vector.size() << std::endl;
        cv::Mat frame_ = frame_vector[0];       
        // std::cout << "图像尺寸: " << frame.cols << "x" << frame.rows << "，通道数: " << frame.channels()<< "，数据类型: " << frame.type() << std::endl;
        cuda_preprocess(
            frame.ptr(),   // 源图像数据指针 (CPU 内存)
            frame.cols,    // 源图像宽度
            frame.rows,    // 源图像高度
            gpu_input_buffer_,        // GPU 输入缓冲区指针
            kInputW,                  // 目标图像宽度 (模型输入尺寸)
            kInputH,                  // 目标图像高度 (模型输入尺寸)
            stream_                   // CUDA 流对象
            );
        std::cout << "图像预处理完成，开始推理..." << std::endl;
        // 准备GPU缓冲区指针数组
        void* buffers[] = {gpu_input_buffer_, gpu_output_buffer_};
        
        // 执行推理
        auto start = std::chrono::high_resolution_clock::now();
        // context_->enqueue(1, buffers, stream_, nullptr);
        context_->setInputTensorAddress(input_tensor_name_, gpu_input_buffer_);
        context_->setTensorAddress(output_tensor_name_, gpu_output_buffer_);
        context_->enqueueV3(stream_);
        std::cout << "推理执行完成，开始从GPU复制结果..." << std::endl;
        // 将结果从GPU异步复制到CPU
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_, gpu_output_buffer_, 
                                  kOutputSize * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_));
        
        // 等待流中所有操作完成
        cudaStreamSynchronize(stream_);
        auto end = std::chrono::high_resolution_clock::now();
        start = std::chrono::high_resolution_clock::now();
        context_->enqueueV3(stream_);
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_, gpu_output_buffer_, 
                                kOutputSize * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_));
        cudaStreamSynchronize(stream_);

        // 检查前20个输出值
        std::cout << "First 20 output values: ";
        for (int i = 0; i < 20; ++i) {
            std::cout << cpu_output_buffer_[i] << " ";
        }
        std::cout << std::endl;
        // 输出推理时间
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "推理时间: " << duration.count() << "ms" << std::endl;
    }

    // TensorRT对象
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    // CUDA对象
    cudaStream_t stream_ = nullptr;
    int input_index_ = -1;                 // 输入张量索引
    int output_index_ = -1;                 // 输出张量索引
    const char* input_tensor_name_ = nullptr;  // 输入张量名称
    const char* output_tensor_name_ = nullptr; // 输出张量名称


    bool enable_imshow = true;  // 是否启用imshow显示检测结果




    // GPU缓冲区
    float* gpu_input_buffer_ = nullptr;
    float* gpu_output_buffer_ = nullptr;
    
    // CPU输出缓冲区
    float* cpu_output_buffer_ = nullptr;
};

class CameraNode : public rclcpp::Node
{
public:
  CameraNode() : Node("tensor_node")
  {
    // 声明配置参数
    declare_parameters();
    // 创建发布者
    create_publishers();
    // 启动深度相机线程
    
    try {
        // 使用参数初始化检测器
        yolo_detector_ = std::make_unique<YoloDetector>(
            0,
            yolo_engine_path_,  // YOLOv5引擎路径
            enable_imshow_
            //yolo_confidence_threshold_,  // YOLOv5置信度阈值
            //yolo_nms_threshold_    // YOLOv5 NMS阈值
        ); 
        RCLCPP_INFO(this->get_logger(), "YOLOv5 TensorRT检测器初始化成功");
    } catch (const std::exception& e) {
          RCLCPP_ERROR(this->get_logger(), "YOLOv5检测器初始化失败: %s", e.what());
    }
    
    depth_thread_ = std::thread(&CameraNode::run_camera, this);
  }
  ~CameraNode() {
    if(depth_thread_.joinable()) {
      depth_thread_.join();
    }
  }
private:
  // 声明参数
  void declare_parameters() 
  {
    // YOLOv5参数
    this->declare_parameter("imshow", true);
    enable_imshow_ = this->get_parameter("imshow").as_bool();
    this->declare_parameter("yolo_engine_path", "weight/yolov5s.engine");
    // yolo_engine_path_ = this->get_parameter("yolo_engine_path").as_string();
    yolo_engine_path_ = "/home/epoch/Desktop/1/R1_vision_v2/src/tensorrt_ros/src/weight/yolov5s.engine";


  }
  // 创建发布者
  void create_publishers() 
  {
    // 自定义消息发布者 (包含所有原始数据)
    msg_data_publisher_ = this->create_publisher<interfaces::msg::Tensorrt>("raw_detection_data", 10);
  }
  // 启动相机线程
  // 使用RealSense SDK获取深度和RGB图像，并进行YOLOv5检测
  // 发布深度信息、检测结果和原始图像
  void run_camera() 
  {
    RCLCPP_INFO(this->get_logger(), "启动深度相机...");
    // 创建管道和配置
    rs2::pipeline p;
    rs2::config cfg;
    // 配置深度流和彩色流
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    // 启动管道
    rs2::pipeline_profile profile = p.start(cfg);
    while (rclcpp::ok())
    {
        rs2::frameset frames = p.wait_for_frames();
        RCLCPP_INFO(this->get_logger(), "启动深度相机...");
        // 获取深度帧
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        float width = depth_frame.get_width();
        float height = depth_frame.get_height();
        // 获取中心距离
        float dist_to_center = depth_frame.get_distance(width / 2, height / 2);
        printf("中心深度: %.2f 米\n", dist_to_center);
        // 获取彩色帧
        rs2::video_frame color_frame = frames.get_color_frame();
        // 转换为OpenCV矩阵
        cv::Mat color_image(cv::Size(width, height), CV_8UC3, 
                           (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);
        // 执行YOLOv5检测
        
        std::vector<Detection> detections;
        try {
            detections= yolo_detector_->detect(color_image);
            printf("各个值是: \n");
            for (const auto& det : detections) {
                printf("检测到目标 - 类别: %d, 置信度: %.2f, 边界框: [%.2f, %.2f, %.2f, %.2f]\n",
                       static_cast<int>(det.class_id), det.conf,
                       det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]);
            }
            RCLCPP_INFO(this->get_logger(), "推理完成，检测到 %zu 个目标", detections.size());
            //publish_detections(detections);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "目标检测失败: %s", e.what());
        }



        // 发布深度信息

        if(enable_imshow_){
        RCLCPP_INFO(this->get_logger(), "显示检测结果...");
        cv::Mat detection_image = draw_detections(color_image, detections);
        }
        // 发布自定义消息 (包含所有原始数据)
        publish_msg_data(color_frame, depth_frame, detections, dist_to_center);
    }
    
    RCLCPP_INFO(this->get_logger(), "深度相机线程退出");
  }

    // 绘制检测结果
    // 在图像上绘制检测框和标签
    // 使用OpenCV绘制边界框和标签
    cv::Mat draw_detections(const cv::Mat& image, const std::vector<Detection>& detections) {
        cv::Mat output_image = image.clone();
        int image_width = output_image.cols;
        int image_height = output_image.rows;
        printf("图像尺寸: %dx%d\n", image_width, image_height);
        // 计算缩放比例
        float ratio_x = static_cast<float>(image_width) / kInputW;
        float ratio_y = static_cast<float>(image_height) / kInputH;
        printf("缩放比例: %.2f, %.2f\n", ratio_x, ratio_y);
        for (const auto& detection : detections) {
            // 边界框坐标转换 (从网络尺寸到原始图像尺寸)
            float x = detection.bbox[0] * ratio_x;
            float y = detection.bbox[1] * ratio_y;
            float w = detection.bbox[2] * ratio_x;
            float h = detection.bbox[3] * ratio_y;
            
            // 创建边界框矩形
            cv::Rect bbox(static_cast<int>(x - w/2), 
                        static_cast<int>(y - h/2),
                        static_cast<int>(w), 
                        static_cast<int>(h));
            
            // 获取类别标签和颜色
            int class_id = static_cast<int>(detection.class_id);
            float confidence = detection.conf;
            
            // 查找预定义颜色，如果没有则为白色
            cv::Scalar color = (class_colors.find(class_id) != class_colors.end()) 
                            ? class_colors.at(class_id) 
                            : cv::Scalar(255, 255, 255);
            
            // 绘制边界框
            cv::rectangle(output_image, bbox, color, 2);
            
            // 创建文本标签
            std::string label = std::to_string(class_id) + ": " + std::to_string(confidence).substr(0, 4);
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
            
            // 绘制文本背景
            cv::rectangle(output_image, 
                        cv::Point(bbox.x, bbox.y - text_size.height - 5),
                        cv::Point(bbox.x + text_size.width, bbox.y),
                        color, cv::FILLED);
            
            // 绘制文本
            cv::putText(output_image, label,
                    cv::Point(bbox.x, bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(0, 0, 0), 1);
        }
        cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
        // 如果启用imshow，则显示检测结果
        cv::imshow("Detections", output_image);
        cv::waitKey(1);  // 显示图像并等待1毫秒
        return output_image;
    }
  
  // 发布自定义消息，包含所有原始数据和检测结果
  // 包括彩色图像、深度图像、检测结果和中心深度
  // 该消息类型为tensorrt_ros::msg::Tensorrt
  void publish_msg_data(const rs2::video_frame& color_frame, 
                        const rs2::depth_frame& depth_frame,
                        const std::vector<Detection>& detections,
                        float center_depth) 
  {
    auto msg = interfaces::msg::Tensorrt();
    
    // 设置时间戳
    msg.header.stamp = this->now();
    msg.header.frame_id = "camera";
    
    // 添加中心深度
    msg.center_depth = center_depth;
    
    // 添加检测结果
    for (const auto& det : detections) {
        msg.x = det.bbox[0];
        msg.y = det.bbox[1];
        msg.width = det.bbox[2];
        msg.height = det.bbox[3];
        msg.class_id = static_cast<int>(det.class_id);
        msg.confidence = det.conf;
    }
    
    msg_data_publisher_->publish(msg);
  }

  // 成员变量
  rclcpp::Publisher<interfaces::msg::Tensorrt>::SharedPtr msg_data_publisher_;
      std::map<int, cv::Scalar> class_colors;  // 类别颜色映射
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_image_publisher_;  // 图像发布器


  std::thread depth_thread_;
  std::unique_ptr<YoloDetector> yolo_detector_;
  
  // 参数
  bool enable_imshow_;
  std::string yolo_engine_path_;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}