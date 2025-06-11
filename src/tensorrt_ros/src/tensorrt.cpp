#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "trt.h"
#include "preprocess.h"
#include "yololayer.h"
#include "postprocess.h"
#include "model.h"

#include "cuda_utils.h"        // CUDA工具函数封装
#include "logging.h"            // TensorRT日志记录
#include "utils.h"              // 通用工具函数（文件读取等）
#include "preprocess.h"         // CUDA预处理函数
#include "postprocess.h"        // 后处理函数（NMS等）
#include "model.h"             // 模型构建相关函数
#include <iostream>             // 输入输出流
#include <chrono>               // 时间测量
#include <cmath>                // 数学函数




#include "types.h"
#include "config.h"


using namespace std::chrono_literals;
using namespace nvinfer1;

// 自定义消息类型
#include "interfaces/msg/tensorrt.hpp"

// TensorRT全局资源
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    float* gpu_buffers[2];  // GPU缓冲区数组（输入/输出）
    float* cpu_output_buffer = nullptr;



// TensorRT日志记录器
static Logger gLogger;
// 输出数据大小计算：最大检测框数量 * 每个检测框大小 / 浮点数大小 + 1（额外信息）
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class CameraNode : public rclcpp::Node
{
public:
  CameraNode() : Node("tensor_node")
  {
    // 声明配置参数
    declare_parameters();
    // 创建发布者
    create_publishers();

    initialize_trt(&runtime, &engine, &context, yolo_engine_path_, gpu_buffers, &cpu_output_buffer);
    // 启动深度相机线程
    depth_thread_ = std::thread(&CameraNode::run_camera, this);
  }
  
  ~CameraNode() {
    if(depth_thread_.joinable()) {
      depth_thread_.join();
    }
        // ------------------ 资源清理 ------------------
    // 销毁CUDA流
    cudaStreamDestroy(stream);
    // 释放GPU输入/输出缓冲区
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    // 释放CPU输出缓冲区
    delete[] cpu_output_buffer;
    // 清理CUDA预处理资源
    cuda_preprocess_destroy();
    
    // 销毁TensorRT对象
    context->destroy();
    engine->destroy();
    runtime->destroy();
  }
  
private:
  // 初始化TensorRT
  void initialize_trt( IRuntime** runtime, ICudaEngine** engine,
                      IExecutionContext** context, const std::string& engine_path, 
                      float* gpu_buffers[2], float** cpu_output_buffer)
  {

    cudaSetDevice(kGpuId);
    // 反序列化模式（推理）------------------------------
    // 加载TensorRT引擎
    deserialize_engine(yolo_engine_path_, runtime, engine, context);
    
    // 创建CUDA流
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 初始化CUDA预处理（分配GPU内存）
    cuda_preprocess_init(kMaxInputImageSize);

    
    prepare_buffers(*engine, &gpu_buffers[0], &gpu_buffers[1], cpu_output_buffer);
  }
  // 声明参数
  void declare_parameters() 
  {
    this->declare_parameter("imshow", true);
    enable_imshow_ = this->get_parameter("imshow").as_bool();
    this->declare_parameter("yolo_engine_path", "weight/yolov5s.engine");
    yolo_engine_path_ = this->get_parameter("yolo_engine_path").as_string();
    yolo_engine_path_ = "/home/epoch/Desktop/1/R1_vision_v2/src/tensorrt_ros/src/weight/yolov5s.engine";
    if (yolo_engine_path_.empty()) {
        yolo_engine_path_ = "/home/epoch/Desktop/1/R1_vision_v2/src/tensorrt_ros/src/weight/yolov5s.engine";
    }
  }
  
  // 创建发布者
  void create_publishers() 
  {
    msg_data_publisher_ = this->create_publisher<interfaces::msg::Tensorrt>("raw_detection_data", 10);
  }
  
  std::vector<Detection> detect(const cv::Mat& image) 
  {


    std::vector<cv::Mat> img_batch = {image};
      // 设置使用的GPU设备



      // CUDA批量预处理（GPU上进行图像缩放、归一化等）
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // 执行推理并计时
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "推理时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "毫秒" << std::endl;

    // 批处理非极大值抑制（过滤重叠框）
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

    // 在图像上绘制检测框
    draw_bbox(img_batch, res_batch);
    
    cv::imshow("Detection Result", img_batch[0]);
    cv::waitKey(1);

    return res_batch[0];  // 返回第一个图像的检测结果


  }

  // 相机主循环
  void run_camera() 
  {
    RCLCPP_INFO(this->get_logger(), "启动深度相机...");
    
    // 创建RealSense管道和配置
    rs2::pipeline p;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    
    // 启动管道
    rs2::pipeline_profile profile = p.start(cfg);
    
    // 获取相机内参（用于3D坐标计算）
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    rs2_intrinsics color_intrin = color_stream.get_intrinsics();
    RCLCPP_INFO(this->get_logger(), "彩色相机内参: fx=%.2f, fy=%.2f, ppx=%.2f, ppy=%.2f",
                color_intrin.fx, color_intrin.fy, color_intrin.ppx, color_intrin.ppy);
    
    // 主循环
    while (rclcpp::ok())
    {
        rs2::frameset frames = p.wait_for_frames();
        
        // 获取深度帧
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        float dist_to_center = depth_frame.get_distance(depth_frame.get_width() / 2, 
                                                      depth_frame.get_height() / 2);
        RCLCPP_INFO(this->get_logger(), "中心深度: %.2f米", dist_to_center);
        
        // 获取彩色帧并转换为OpenCV格式
        rs2::video_frame color_frame = frames.get_color_frame();
        cv::Mat color_image(cv::Size(640, 480), CV_8UC3, 
                           (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        // cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);
        // 执行目标检测
        std::vector<Detection> detections;
        try {
            detections = detect(color_image);
            RCLCPP_INFO(this->get_logger(), "检测到 %zu 个目标", detections.size());
            
            // // 计算3D坐标
            // for (const auto& det : detections) {
            //     // 边界框中心
            //     float center_x = det.bbox[0] + det.bbox[2] / 2;
            //     float center_y = det.bbox[1] + det.bbox[3] / 2;
                
            //     // 获取深度值
            //     float depth = depth_frame.get_distance(center_x, center_y);
                
            //     // 转换到3D坐标
            //     float pixel[2] = {center_x, center_y};
            //     float point[3];
            //     rs2_deproject_pixel_to_point(point, &color_intrin, pixel, depth);
                
            //     RCLCPP_INFO(this->get_logger(), 
            //                 "目标 %d @ (%.2f, %.2f, %.2f) | 置信度: %.2f", 
            //                 det.class_id, point[0], point[1], point[2], det.conf);
            // }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "目标检测失败: %s", e.what());
        }

        // 显示检测结果
        // if (enable_imshow_ && !detections.empty()) {
        //     cv::Mat output_image = draw_detections(color_image, detections);
        //     cv::imshow("RealSense Detection", output_image);
        //     cv::waitKey(1);
        // }
        
        // 发布自定义消息
        publish_msg_data(detections, dist_to_center);
    }
    
    RCLCPP_INFO(this->get_logger(), "深度相机线程退出");
  }

  // 绘制检测结果
  cv::Mat draw_detections(const cv::Mat& image, const std::vector<Detection>& detections) {
      cv::Mat output_image = image.clone();
      
      for (const auto& detection : detections) {
          // 提取边界框信息
          float x = detection.bbox[0];
          float y = detection.bbox[1];
          float w = detection.bbox[2];
          float h = detection.bbox[3];
          
          cv::Rect bbox(x, y, w, h);
          
          // 绘制边界框
          cv::rectangle(output_image, bbox, cv::Scalar(0, 255, 0), 2);
          
          // 创建文本标签
          std::string label = "ID:" + std::to_string(detection.class_id) + 
                            " Conf:" + std::to_string(detection.conf).substr(0, 4);
          cv::putText(output_image, label, cv::Point(bbox.x, bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
      }
      
      return output_image;
  }
  
  // 发布自定义消息
  void publish_msg_data(
                      const std::vector<Detection>& detections,
                      float center_depth) 
  {
    auto msg = interfaces::msg::Tensorrt();
    msg.header.stamp = this->now();
    msg.header.frame_id = "camera";
    msg.center_depth = center_depth;
    
    for (const auto& det : detections) {
        msg.x=det.bbox[0];
        msg.y=det.bbox[1];
        msg.width=det.bbox[2];
        msg.height=det.bbox[3];
        msg.class_id=det.class_id;
        msg.confidence=det.conf;
    }
    
    msg_data_publisher_->publish(msg);
  }

  // 成员变量
  rclcpp::Publisher<interfaces::msg::Tensorrt>::SharedPtr msg_data_publisher_;
  std::thread depth_thread_;
  
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