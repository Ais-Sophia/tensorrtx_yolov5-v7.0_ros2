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
  // 获取相机外参（深度到彩色）
  std::pair<cv::Mat, cv::Mat> get_depth_to_color_extrinsics(rs2::pipeline& pipe) {
      auto frames = pipe.wait_for_frames();
      auto depth_frame = frames.get_depth_frame();
      auto color_frame = frames.get_color_frame();
      
      auto depth_profile = depth_frame.get_profile().as<rs2::video_stream_profile>();
      auto color_profile = color_frame.get_profile().as<rs2::video_stream_profile>();
      
      rs2_extrinsics extrinsics = depth_profile.get_extrinsics_to(color_profile);
      
      // 转换为OpenCV矩阵
      cv::Mat R = (cv::Mat_<float>(3, 3) << 
          extrinsics.rotation[0], extrinsics.rotation[1], extrinsics.rotation[2],
          extrinsics.rotation[3], extrinsics.rotation[4], extrinsics.rotation[5],
          extrinsics.rotation[6], extrinsics.rotation[7], extrinsics.rotation[8]);
      
      cv::Mat t = (cv::Mat_<float>(3, 1) << 
          extrinsics.translation[0], 
          extrinsics.translation[1], 
          extrinsics.translation[2]);
      
      return {R, t};
  }
  // 彩色像素坐标转深度点云坐标
  cv::Point3f color_pixel_to_depth_point(
      int u_color, int v_color, 
      const rs2::depth_frame& depth_frame,
      const cv::Mat& camera_matrix_color,
      const cv::Mat& camera_matrix_depth,
      const cv::Mat& R, const cv::Mat& t) 
  {
      // 提取内参参数
      float fx_color = camera_matrix_color.at<float>(0, 0);
      float fy_color = camera_matrix_color.at<float>(1, 1);
      float cx_color = camera_matrix_color.at<float>(0, 2);
      float cy_color = camera_matrix_color.at<float>(1, 2);
      
      float fx_depth = camera_matrix_depth.at<float>(0, 0);
      float fy_depth = camera_matrix_depth.at<float>(1, 1);
      float cx_depth = camera_matrix_depth.at<float>(0, 2);
      float cy_depth = camera_matrix_depth.at<float>(1, 2);
      
      // 步骤1：获取初始深度值（取深度图中心点）
      int center_u = depth_frame.get_width() / 2;
      int center_v = depth_frame.get_height() / 2;
      float z0 = depth_frame.get_distance(center_u, center_v);
      if (z0 <= 0) z0 = 1.0f; // 无效深度时使用默认值
      
      // 步骤2：彩色像素坐标转彩色相机坐标系
      float Xc = (u_color - cx_color) * z0 / fx_color;
      float Yc = (v_color - cy_color) * z0 / fy_color;
      float Zc = z0;
      
      // 步骤3：坐标转换到深度相机坐标系
      cv::Mat Pc = (cv::Mat_<float>(3, 1) << Xc, Yc, Zc);
      cv::Mat R_inv = R.t();  // 旋转矩阵正交，转置=逆
      cv::Mat Pd = R_inv * (Pc - t);
      
      // 步骤4：投影到深度图像素坐标
      float Xd = Pd.at<float>(0);
      float Yd = Pd.at<float>(1);
      float Zd = Pd.at<float>(2);
      int u_depth = static_cast<int>(fx_depth * (Xd / Zd) + cx_depth);
      int v_depth = static_cast<int>(fy_depth * (Yd / Zd) + cy_depth);
      
      // 步骤5：获取精确深度值
      float d_final = depth_frame.get_distance(u_depth, v_depth);
      
      // 无效深度处理策略
      const float MAX_DEPTH = 10.0f; // 最大有效深度(米)
      if (d_final <= 0 || d_final > MAX_DEPTH) {
          // 策略1：使用估计的深度值
          // return cv::Point3f(Xd, Yd, Zd);
          
          // 策略2：取周围区域的深度中值
          const int KERNEL_SIZE = 5;
          std::vector<float> depths;
          
          for (int du = -KERNEL_SIZE/2; du <= KERNEL_SIZE/2; du++) {
              for (int dv = -KERNEL_SIZE/2; dv <= KERNEL_SIZE/2; dv++) {
                  int x = u_depth + du;
                  int y = v_depth + dv;
                  if (x >= 0 && x < depth_frame.get_width() && 
                      y >= 0 && y < depth_frame.get_height()) {
                      float d = depth_frame.get_distance(x, y);
                      if (d > 0 && d <= MAX_DEPTH) {
                          depths.push_back(d);
                      }
                  }
              }
          }
          
          if (!depths.empty()) {
              std::sort(depths.begin(), depths.end());
              d_final = depths[depths.size() / 2]; // 中值
          } else {
              return cv::Point3f(0, 0, 0); // 返回无效点
          }
      }
      
      // 步骤6：计算最终深度坐标
      float Xf = (u_depth - cx_depth) * d_final / fx_depth;
      float Yf = (v_depth - cy_depth) * d_final / fy_depth;
      float Zf = d_final;
      
      return cv::Point3f(Xf, Yf, Zf);
  }

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
    yolo_engine_path_ = "/home/epoch/tensorrtx_ros2/src/tensorrt_ros/src/weight/volleyball.engine";
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
    auto start = std::chrono::system_clock::now();
    RCLCPP_INFO(this->get_logger(), "启动深度相机...");
    rs2::pipeline p;
    rs2::config cfg;
    // 创建RealSense管道和配置
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    rs2::pipeline_profile profile = p.start(cfg);
    auto end = std::chrono::system_clock::now();
    std::cout << "相机初始化时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "毫秒" << std::endl;
    // 主循环
    while (rclcpp::ok())
    {   
        rs2::frameset frames;
        frames = p.wait_for_frames();  // 设置超时时间(ms
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        rs2::video_frame color_frame = frames.get_color_frame();
        // 获取内外参数
        auto [R, t] = get_depth_to_color_extrinsics(p);
        auto color_profile = color_frame.get_profile().as<rs2::video_stream_profile>();
        auto color_intrin = color_profile.get_intrinsics();
        cv::Mat camera_matrix_color = (cv::Mat_<float>(3, 3) << 
            color_intrin.fx, 0, color_intrin.ppx,
            0, color_intrin.fy, color_intrin.ppy,
            0, 0, 1);
        auto depth_profile = depth_frame.get_profile().as<rs2::video_stream_profile>();
        auto depth_intrin = depth_profile.get_intrinsics();
        cv::Mat camera_matrix_depth = (cv::Mat_<float>(3, 3) << 
            depth_intrin.fx, 0, depth_intrin.ppx,
            0, depth_intrin.fy, depth_intrin.ppy,
            0, 0, 1);
        float dist_to_center = depth_frame.get_distance(depth_frame.get_width() / 2, 
                                                      depth_frame.get_height() / 2);
        cv::Mat color_image(cv::Size(640, 480), CV_8UC3, 
                           (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        // 执行目标检测
        float x=0.0,y=0.0;
        float depth_value = 0.0 ;
        float pixel[2] = {0.0, 0.0};
        // 将像素坐标投影到3D坐标
        float point[3]={0.0, 0.0, 0.0};
        cv::Point3f point3d(0.0, 0.0, 0.0);
        std::vector<Detection> detections;
        try {
            detections = detect(color_image);
            RCLCPP_INFO(this->get_logger(), "检测到 %zu 个目标", detections.size());
            // // 计算3D坐标
            //     // 边界框中心
            //计算边界框中心坐标哦
           if (!detections.empty()) {
              x = (detections[0].bbox[0] + detections[0].bbox[2]) / 2;
              y = (detections[0].bbox[1] + detections[0].bbox[3]) / 2;
              // ... [使用x,y]
              depth_value = depth_frame.get_distance(
                        static_cast<int>(std::round(x)), 
                        static_cast<int>(std::round(y))
                    );
              point3d = color_pixel_to_depth_point(
                  (int)x, (int)y, depth_frame,
                  camera_matrix_color, camera_matrix_depth,
                  R, t
              );
              // pixel[0] = x;
              // pixel[1] = y;
              // rs2_deproject_pixel_to_point(point, &depth_intrin, pixel, depth_value);
            } else {
              RCLCPP_INFO(this->get_logger(), "无目标，跳过3D计算");
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "目标检测失败: %s", e.what());
        }
        // 发布自定义消息
        publish_msg_data(detections, dist_to_center, &point3d);
    }
    
    RCLCPP_INFO(this->get_logger(), " 深度相机线程退出");
  }
  // 发布自定义消息
  void publish_msg_data(
                      const std::vector<Detection>& detections,
                      float center_depth,
                      cv::Point3f* point) 
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
    msg.point[0] = point->x;//右
    msg.point[1] = point->y;//下
    msg.point[2] = point->z;//前
    // 发布消息
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