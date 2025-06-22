#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <iostream>

// 自定义常量
const int COLOR_WIDTH = 640;
const int COLOR_HEIGHT = 480;
const int DEPTH_WIDTH = 640;
const int DEPTH_HEIGHT = 480;

// 相机内参结构体
struct CameraIntrinsics {
    float fx, fy;     // 焦距
    float ppx, ppy;   // 主点坐标
    float coeffs[5];  // 畸变系数
};

// 相机外参结构体
struct CameraExtrinsics {
    float rotation[9];   // 旋转矩阵
    float translation[3]; // 平移向量
};

// 转换深度图到点云
__global__ void depth_to_points_kernel(
    const uint16_t* depth_data, 
    float3* points, 
    int width, 
    int height, 
    const CameraIntrinsics intrinsics)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (u < width && v < height) {
        int idx = v * width + u;
        uint16_t depth_value = depth_data[idx];
        
        if (depth_value == 0) {
            points[idx] = make_float3(0, 0, 0);
            return;
        }
        
        float z = depth_value * 0.001f;  // 转换为米
        float x = (u - intrinsics.ppx) * z / intrinsics.fx;
        float y = (v - intrinsics.ppy) * z / intrinsics.fy;
        
        // 考虑径向畸变
        float x_dist = u / fx;
        float y_dist = v / fy;
        float r2 = x_dist*x_dist + y_dist*y_dist;
        float radial_dist = 1.0f + intrinsics.coeffs[0] * r2 + intrinsics.coeffs[1] * r2*r2;
        
        points[idx] = make_float3(x * radial_dist, y * radial_dist, z);
    }
}

// 对齐深度到彩色图坐标系的核函数
__global__ void align_depth_to_color_kernel(
    const float3* depth_points, 
    uint16_t* aligned_depth, 
    const CameraExtrinsics extrinsics, 
    const CameraIntrinsics color_intr,
    int depth_width, 
    int depth_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < COLOR_WIDTH && y < COLOR_HEIGHT) {
        // 1. 对于彩色图像上的每个像素，计算对应的深度图坐标
        float2 dist_pixel = {(x - color_intr.ppx) / color_intr.fx, 
                             (y - color_intr.ppy) / color_intr.fy};
        
        // 2. 应用畸变校正
        float r2 = dist_pixel.x*dist_pixel.x + dist_pixel.y*dist_pixel.y;
        float f = 1.0f + color_intr.coeffs[0]*r2 + color_intr.coeffs[1]*r2*r2;
        float x_dist = dist_pixel.x * f;
        float y_dist = dist_pixel.y * f;
        
        // 3. 转换到深度相机坐标系
        float3 color_point = make_float3(x_dist, y_dist, 1.0f);
        float3 depth_point = {
            extrinsics.rotation[0]*color_point.x + extrinsics.rotation[1]*color_point.y + extrinsics.rotation[2]*color_point.z + extrinsics.translation[0],
            extrinsics.rotation[3]*color_point.x + extrinsics.rotation[4]*color_point.y + extrinsics.rotation[5]*color_point.z + extrinsics.translation[1],
            extrinsics.rotation[6]*color_point.x + extrinsics.rotation[7]*color_point.y + extrinsics.rotation[8]*color_point.z + extrinsics.translation[2]
        };
        
        // 4. 投影到深度图像
        int u_depth = (int)roundf(depth_point.x * depth_intr.fx / depth_point.z + depth_intr.ppx);
        int v_depth = (int)roundf(depth_point.y * depth_intr.fy / depth_point.z + depth_intr.ppy);
        
        // 5. 获取深度值
        uint16_t depth_value = 0;
        if (u_depth >= 0 && u_depth < depth_width && v_depth >= 0 && v_depth < depth_height) {
            int depth_idx = v_depth * depth_width + u_depth;
            depth_value = depth_points[depth_idx].z * 1000;  // 转回毫米单位
        }
        
        // 6. 保存对齐后的深度值
        aligned_depth[y * COLOR_WIDTH + x] = depth_value;
    }
}

// 计算边界框中心深度值的核函数
__global__ void compute_box_depth_kernel(
    const uint16_t* aligned_depth,
    const float* bboxes,   // [x, y, w, h, class_id, confidence]
    float* box_depths,
    int num_boxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        int offset = idx * 6;  // 每个检测框6个元素
        
        float x = bboxes[offset];
        float y = bboxes[offset + 1];
        float w = bboxes[offset + 2];
        float h = bboxes[offset + 3];
        
        // 计算中心坐标
        int center_x = static_cast<int>(x + w / 2.0f);
        int center_y = static_cast<int>(y + h / 2.0f);
        
        // 获取深度值（单位米）
        float depth_value = 0.0f;
        if (center_x >= 0 && center_x < COLOR_WIDTH && center_y >= 0 && center_y < COLOR_HEIGHT) {
            depth_value = aligned_depth[center_y * COLOR_WIDTH + center_x] * 0.001f;
        }
        
        box_depths[idx] = depth_value;
    }
}

// 计算边界框平均深度的核函数
__global__ void compute_box_avg_depth_kernel(
    const uint16_t* aligned_depth,
    const float* bboxes,   // [x, y, w, h, class_id, confidence]
    float* box_avg_depths,
    int num_boxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        int offset = idx * 6;  // 每个检测框6个元素
        
        float x = bboxes[offset];
        float y = bboxes[offset + 1];
        float w = bboxes[offset + 2];
        float h = bboxes[offset + 3];
        
        // 计算边界坐标
        int x1 = max(0, min(COLOR_WIDTH - 1, static_cast<int>(x)));
        int y1 = max(0, min(COLOR_HEIGHT - 1, static_cast<int>(y)));
        int x2 = max(0, min(COLOR_WIDTH - 1, static_cast<int>(x + w)));
        int y2 = max(0, min(COLOR_HEIGHT - 1, static_cast<int>(y + h)));
        
        // 计算边界框内有效深度的平均值
        float sum = 0.0f;
        int count = 0;
        
        for (int iy = y1; iy <= y2; iy++) {
            for (int ix = x1; ix <= x2; ix++) {
                uint16_t depth_value = aligned_depth[iy * COLOR_WIDTH + ix];
                if (depth_value > 0) {  // 忽略无效深度
                    sum += depth_value * 0.001f;  // 转换为米
                    count++;
                }
            }
        }
        
        box_avg_depths[idx] = (count > 0) ? (sum / count) : 0.0f;
    }
}

// CUDA包装函数
extern "C" {

void cuda_align_depth_to_color(
    const uint16_t* depth_data,
    uint16_t* aligned_depth,
    const CameraIntrinsics depth_intr,
    const CameraIntrinsics color_intr,
    const CameraExtrinsics extrinsics)
{
    // 1. 转换深度图到点云
    float3* d_depth_points;
    cudaMalloc(&d_depth_points, DEPTH_WIDTH * DEPTH_HEIGHT * sizeof(float3));
    
    dim3 block(16, 16);
    dim3 grid((DEPTH_WIDTH + block.x - 1) / block.x, (DEPTH_HEIGHT + block.y - 1) / block.y);
    
    depth_to_points_kernel<<<grid, block>>>(depth_data, d_depth_points, 
                                         DEPTH_WIDTH, DEPTH_HEIGHT, depth_intr);
    
    // 2. 执行深度对齐
    dim3 align_block(16, 16);
    dim3 align_grid((COLOR_WIDTH + align_block.x - 1) / align_block.x, 
                    (COLOR_HEIGHT + align_block.y - 1) / align_block.y);
    
    align_depth_to_color_kernel<<<align_grid, align_block>>>(d_depth_points, aligned_depth, 
                                                          extrinsics, color_intr,
                                                          DEPTH_WIDTH, DEPTH_HEIGHT);
    
    // 清理
    cudaFree(d_depth_points);
    cudaDeviceSynchronize();
}

void cuda_compute_detection_depths(
    const uint16_t* aligned_depth,
    const float* d_bboxes, // GPU上的边界框数据
    float* d_box_depths,   // GPU上的输出深度数组
    int num_boxes,
    bool use_average)
{
    dim3 block(256);
    dim3 grid((num_boxes + block.x - 1) / block.x);
    
    if (use_average) {
        compute_box_avg_depth_kernel<<<grid, block>>>(aligned_depth, d_bboxes, d_box_depths, num_boxes);
    } else {
        compute_box_depth_kernel<<<grid, block>>>(aligned_depth, d_bboxes, d_box_depths, num_boxes);
    }
    
    cudaDeviceSynchronize();
}

}