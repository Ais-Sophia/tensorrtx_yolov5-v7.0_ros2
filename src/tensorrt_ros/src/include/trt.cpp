#include "cuda_utils.h"        // CUDA工具函数封装
#include "logging.h"            // TensorRT日志记录
#include "utils.h"              // 通用工具函数（文件读取等）
#include "preprocess.h"         // CUDA预处理函数
#include "postprocess.h"        // 后处理函数（NMS等）
#include "model.h"             // 模型构建相关函数
#include "trt.h"
#include <iostream>             // 输入输出流
#include <chrono>               // 时间测量
#include <cmath>                // 数学函数

using namespace nvinfer1;      // TensorRT命名空间

// TensorRT日志记录器
static Logger gLogger;
// 输出数据大小计算：最大检测框数量 * 每个检测框大小 / 浮点数大小 + 1（额外信息）
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

/**
 * 解析命令行参数
 * 
 * @param argc 参数数量
 * @param argv 参数值数组
 * @param wts 权重文件路径（输出参数）
 * @param engine 引擎文件路径（输出参数）
 * @param is_p6 是否使用P6模型标志（输出参数）
 * @param gd 模型深度缩放因子（输出参数）
 * @param gw 模型宽度缩放因子（输出参数）
 * @param img_dir 图像目录路径（输出参数）
 * @return 是否解析成功
 */
bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
  // 参数不足时返回解析失败
  if (argc < 4) return false;
  
  // 序列化模式（将.wts转换为.engine）
  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    wts = std::string(argv[2]);       // 权重文件路径
    engine = std::string(argv[3]);    // 输出引擎文件路径
    auto net = std::string(argv[4]);  // 模型类型参数
    
    // 根据模型类型设置不同的缩放因子
    if (net[0] == 'n') {       // nano版模型
      gd = 0.33;
      gw = 0.25;
    } else if (net[0] == 's') { // small版模型
      gd = 0.33;
      gw = 0.50;
    } else if (net[0] == 'm') { // medium版模型
      gd = 0.67;
      gw = 0.75;
    } else if (net[0] == 'l') { // large版模型
      gd = 1.0;
      gw = 1.0;
    } else if (net[0] == 'x') { // xlarge版模型
      gd = 1.33;
      gw = 1.25;
    } else if (net[0] == 'c' && argc == 7) { // 自定义模型
      gd = atof(argv[5]);  // 自定义深度因子
      gw = atof(argv[6]);  // 自定义宽度因子
    } else {
      return false;  // 无效参数
    }
    
    // 检查是否使用P6模型（名字第二个字符为6）
    if (net.size() == 2 && net[1] == '6') {
      is_p6 = true;
    }
  } 
  // 反序列化模式（使用.engine进行推理）
  else if (std::string(argv[1]) == "-d" && argc == 4) {
    engine = std::string(argv[2]);  // 引擎文件路径
    img_dir = std::string(argv[3]); // 图像目录路径
  } else {
    return false;  // 无效参数
  }
  return true;  // 解析成功
}

/**
 * 准备GPU输入/输出缓冲区和CPU输出缓冲区
 * 
 * @param engine TensorRT引擎对象
 * @param gpu_input_buffer GPU输入缓冲区指针（输出）
 * @param gpu_output_buffer GPU输出缓冲区指针（输出）
 * @param cpu_output_buffer CPU输出缓冲区指针（输出）
 */
void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  // 确保引擎有两个binding（输入+输出）
  assert(engine->getNbBindings() == 2);
  
  // 获取输入输出Tensor的索引
  const int inputIndex = engine->getBindingIndex(kInputTensorName);   // 输入Tensor索引
  const int outputIndex = engine->getBindingIndex(kOutputTensorName); // 输出Tensor索引
  
  // 验证索引正确性
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  
  // 在GPU设备上分配输入缓冲区内存（批大小×通道×高度×宽度）
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  // 在GPU设备上分配输出缓冲区内存
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));
  
  // 在主机(CPU)分配输出缓冲区内存
  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

/**
 * 执行推理
 * 
 * @param context TensorRT执行上下文
 * @param stream CUDA流对象
 * @param gpu_buffers GPU缓冲区数组
 * @param output CPU输出缓冲区
 * @param batchsize 批量大小
 */
void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
  // 执行异步推理（将任务加入CUDA流）
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  
  // 将结果从GPU异步复制到CPU
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), 
                            cudaMemcpyDeviceToHost, stream));
  // 等待流中所有操作完成
  cudaStreamSynchronize(stream);
}

/**
 * 序列化TensorRT引擎
 * 
 * @param max_batchsize 最大批大小
 * @param is_p6 是否使用P6模型
 * @param gd 模型深度缩放因子
 * @param gw 模型宽度缩放因子
 * @param wts_name 权重文件名
 * @param engine_name 引擎文件名
 */
void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // 创建推理构建器
  IBuilder* builder = createInferBuilder(gLogger);
  // 创建构建器配置
  IBuilderConfig* config = builder->createBuilderConfig();

  // 构建CUDA引擎（区分P5和P6模型）
  ICudaEngine *engine = nullptr;
  if (is_p6) {
    engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  } else {
    engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  }
  assert(engine != nullptr);  // 确保引擎创建成功

  // 序列化引擎
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);  // 确保序列化成功

  // 保存序列化后的引擎到文件
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "无法打开引擎输出文件" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // 清理资源
  engine->destroy();
  config->destroy();
  serialized_engine->destroy();
  builder->destroy();
}

/**
 * 从文件反序列化TensorRT引擎
 * 
 * @param engine_name 引擎文件名
 * @param runtime TensorRT运行时指针（输出）
 * @param engine TensorRT引擎指针（输出）
 * @param context TensorRT执行上下文指针（输出）
 */
void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  // 打开引擎文件
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "读取" << engine_name << "错误!" << std::endl;
    assert(false);
  }
  
  // 获取文件大小
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  
  // 分配缓冲区读取序列化引擎
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();
  
  // 创建运行时
  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  
  // 反序列化引擎
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  
  // 创建执行上下文
  *context = (*engine)->createExecutionContext();
  assert(*context);
  
  // 清理临时缓冲区
  delete[] serialized_engine;
}

// int main(int argc, char** argv) {
//   // 设置使用的GPU设备
//   cudaSetDevice(kGpuId);

//   // 参数变量初始化
//   std::string wts_name = "/home/epoch/1/volleyball/tensorrtx_ros2/src/tensorrt_ros/src/weight/yolov5s.wts";        // 权重文件路径
//   std::string engine_name = "/home/epoch/1/volleyball/tensorrtx_ros2/src/tensorrt_ros/src/weight/yolov5s_volleyball.engine";     // 引擎文件路径
//   bool is_p6 = false;               // 是否P6模型
//   float gd = 0.33f, gw = 0.50f;       // 模型缩放因子
//   std::string img_dir;              // 图像目录路径

//   // // 解析命令行参数
//   // if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
//   //   std::cerr << "参数不正确!" << std::endl;
//   //   std::cerr << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 或 c/c6 gd gw]  // 序列化模型到引擎文件" << std::endl;
//   //   std::cerr << "./yolov5_det -d [.engine] ../images  // 从引擎文件反序列化并推理" << std::endl;
//   //   return -1;  // 退出程序
//   // }

//   // 序列化模式（创建引擎文件）
//   if (!wts_name.empty()) {
//     serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
//     return 0;
//   }

//   // // 反序列化模式（推理）------------------------------

//   // // 加载TensorRT引擎
//   // IRuntime* runtime = nullptr;
//   // ICudaEngine* engine = nullptr;
//   // IExecutionContext* context = nullptr;
//   // deserialize_engine(engine_name, &runtime, &engine, &context);
  
//   // // 创建CUDA流
//   // cudaStream_t stream;
//   // CUDA_CHECK(cudaStreamCreate(&stream));

//   // // 初始化CUDA预处理（分配GPU内存）
//   // cuda_preprocess_init(kMaxInputImageSize);

//   // // 准备CPU/GPU缓冲区
//   // float* gpu_buffers[2];  // GPU缓冲区数组（输入/输出）
//   // float* cpu_output_buffer = nullptr;  // CPU输出缓冲区
//   // prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

//   // // 读取目录中的图像文件名
//   // std::vector<std::string> file_names;
//   // if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
//   //   std::cerr << "读取目录中的文件失败" << std::endl;
//   //   return -1;
//   // }

//   // // 批处理预测循环
//   // for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
//   //   // 获取一批图像
//   //   std::vector<cv::Mat> img_batch;
//   //   std::vector<std::string> img_name_batch;
//   //   for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
//   //     cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
//   //     img_batch.push_back(img);
//   //     img_name_batch.push_back(file_names[j]);
//   //   }

//   //   // CUDA批量预处理（GPU上进行图像缩放、归一化等）
//   //   cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

//   //   // 执行推理并计时
//   //   auto start = std::chrono::system_clock::now();
//   //   infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
//   //   auto end = std::chrono::system_clock::now();
//   //   std::cout << "推理时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "毫秒" << std::endl;

//   //   // 批处理非极大值抑制（过滤重叠框）
//   //   std::vector<std::vector<Detection>> res_batch;
//   //   batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

//   //   // 在图像上绘制检测框
//   //   draw_bbox(img_batch, res_batch);

//   //   // 保存结果图像（在原文件名前加下划线）
//   //   for (size_t j = 0; j < img_batch.size(); j++) {
//   //     cv::imwrite("_" + img_name_batch[j], img_batch[j]);
//   //   }
//   // }

//   // // ------------------ 资源清理 ------------------
//   // // 销毁CUDA流
//   // cudaStreamDestroy(stream);
//   // // 释放GPU输入/输出缓冲区
//   // CUDA_CHECK(cudaFree(gpu_buffers[0]));
//   // CUDA_CHECK(cudaFree(gpu_buffers[1]));
//   // // 释放CPU输出缓冲区
//   // delete[] cpu_output_buffer;
//   // // 清理CUDA预处理资源
//   // cuda_preprocess_destroy();
  
//   // // 销毁TensorRT对象
//   // context->destroy();
//   // engine->destroy();
//   // runtime->destroy();

//   return 0;  // 程序正常结束
// }