#include <librealsense2/rs.hpp>  // 包含RealSense库头文件

void run_depth_camera() {
    // 创建管道（Pipeline）- 这是用于流式传输和处理帧的顶级API
    rs2::pipeline p;

    // 配置并启动管道
    p.start();

    // 无限循环，持续处理帧数据
    while (true)
    {
        // 阻塞程序直到帧数据到达
        rs2::frameset frames = p.wait_for_frames();

        // 尝试获取深度图像帧
        rs2::depth_frame depth = frames.get_depth_frame();

        // 获取深度帧的尺寸（宽度和高度）
        float width = depth.get_width();
        float height = depth.get_height();

        // 查询相机到图像中心物体的距离
        // 使用图像中心坐标(width/2, height/2)
        float dist_to_center = depth.get_distance(width / 2, height / 2);


}