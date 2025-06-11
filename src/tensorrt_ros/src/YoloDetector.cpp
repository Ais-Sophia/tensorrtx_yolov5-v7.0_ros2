#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

// 常量定义
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class YoloDetector {
public:
    /**
     * 构造函数
     * 
     * @param gpu_id 使用的GPU设备ID
     */
    explicit YoloDetector(int gpu_id = 0) {
        cudaSetDevice(gpu_id);
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
        preprocessAndInference(frame);
        
        // 后处理
        std::vector<Detection> detections;
        nms(detections, cpu_output_buffer_, kOutputSize, kConfThresh, kNmsThresh);
        
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
            context_->destroy();
        }
        if (engine_) {
            engine_->destroy();
        }
        if (runtime_) {
            runtime_->destroy();
        }
    }

private:
    /**
     * 准备GPU/CPU缓冲区
     */
    void prepareBuffers() {
        // 确保引擎有两个绑定(输入+输出)
        assert(engine_->getNbBindings() == 2);
        
        // 获取输入输出Tensor的索引
        const int inputIndex = engine_->getBindingIndex(kInputTensorName);
        const int outputIndex = engine_->getBindingIndex(kOutputTensorName);
        
        // 在GPU设备上分配输入缓冲区内存(1x3xHxW)
        CUDA_CHECK(cudaMalloc((void**)&gpu_input_buffer_, 1 * 3 * kInputH * kInputW * sizeof(float)));
        
        // 在GPU设备上分配输出缓冲区内存
        CUDA_CHECK(cudaMalloc((void**)&gpu_output_buffer_, 1 * kOutputSize * sizeof(float)));
        
        // 在主机(CPU)分配输出缓冲区内存
        cpu_output_buffer_ = new float[1 * kOutputSize];
    }

    /**
     * 图像预处理并执行推理
     * 
     * @param frame 输入图像帧
     */
    void preprocessAndInference(const cv::Mat& frame) {
        // 创建单元素的Mat向量
        std::vector<cv::Mat> frame_vector{frame};
        
        // CUDA预处理(将图像调整大小、归一化并复制到GPU)
        cuda_batch_preprocess(frame_vector, gpu_input_buffer_, kInputW, kInputH, stream_);
        
        // 准备GPU缓冲区指针数组
        void* buffers[] = {gpu_input_buffer_, gpu_output_buffer_};
        
        // 执行推理
        auto start = std::chrono::high_resolution_clock::now();
        context_->enqueue(1, buffers, stream_, nullptr);
        
        // 将结果从GPU异步复制到CPU
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_, gpu_output_buffer_, 
                                  kOutputSize * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_));
        
        // 等待流中所有操作完成
        cudaStreamSynchronize(stream_);
        auto end = std::chrono::high_resolution_clock::now();
        
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
    
    // GPU缓冲区
    float* gpu_input_buffer_ = nullptr;
    float* gpu_output_buffer_ = nullptr;
    
    // CPU输出缓冲区
    float* cpu_output_buffer_ = nullptr;
};