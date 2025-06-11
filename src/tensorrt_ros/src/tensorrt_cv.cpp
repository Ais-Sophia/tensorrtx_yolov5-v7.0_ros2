#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32.hpp"
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp> // 添加OpenCV支持

using namespace std::chrono_literals;

class CameraNode : public rclcpp::Node
{
public:
  CameraNode() : Node("depth_camera_node"), count_(0)
  {
    // 声明是否启用OpenCV显示的参数
    this->declare_parameter("use_opencv", true); // 默认启用显示
    use_opencv_ = this->get_parameter("use_opencv").as_bool();
    
    RCLCPP_INFO(this->get_logger(), "OpenCV显示: %s", use_opencv_ ? "启用" : "禁用");

    // 创建文本发布者
    text_publisher_ = this->create_publisher<std_msgs::msg::String>("text_topic", 10);
    
    // 创建深度数据发布者
    depth_publisher_ = this->create_publisher<std_msgs::msg::Float32>("center_depth_topic", 10);

    // 启动深度相机线程
    depth_thread_ = std::thread(&CameraNode::run_depth_camera, this);
  }
  
  ~CameraNode() {
    // 确保线程安全退出
    if(depth_thread_.joinable()) {
      depth_thread_.join();
    }
    
  }

private:
  
  // 深度相机处理函数
  void run_depth_camera() 
  {
    RCLCPP_INFO(this->get_logger(), "正在启动深度相机...");
    
    // 创建管道（Pipeline）
    rs2::pipeline p;
    rs2::config cfg;
    
    // 配置深度流
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    
    // 启动管道
    rs2::pipeline_profile profile = p.start(cfg);

    // 如果启用OpenCV显示，创建窗口

    while (rclcpp::ok())
    {
        rs2::frameset frames = p.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame();

        float width = depth.get_width();
        float height = depth.get_height();
        float dist_to_center = depth.get_distance(width / 2, height / 2);

        // 创建深度消息并发布
        auto depth_msg = std_msgs::msg::Float32();
        depth_msg.data = dist_to_center;
        depth_publisher_->publish(depth_msg);
        
        // 限流日志输出
        RCLCPP_INFO_THROTTLE(
          this->get_logger(), 
          *this->get_clock(),
          1000,
          "检测到前方物体距离: %.2f 米", 
          dist_to_center);
          
    }
    
    RCLCPP_INFO(this->get_logger(), "深度相机线程安全退出");
  }
  
  // 深度图可视化处理函数
  cv::Mat get_depth_visualization(const rs2::depth_frame& depth_frame) {
    // 获取深度图尺寸
    const int width = depth_frame.get_width();
    const int height = depth_frame.get_height();
    
    // 创建OpenCV矩阵存储深度数据
    cv::Mat depth_image(cv::Size(width, height), CV_8UC3);
    
    // 创建彩色映射深度图
    rs2::colorizer color_map;
    rs2::frame color_depth = color_map.process(depth_frame);
    
    // 转换为OpenCV格式
    depth_image = cv::Mat(cv::Size(width, height), CV_8UC3, 
                         (void*)color_depth.get_data(), cv::Mat::AUTO_STEP);
    
    return depth_image;
  }

  // 成员变量
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr text_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr depth_publisher_;
  std::thread depth_thread_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}