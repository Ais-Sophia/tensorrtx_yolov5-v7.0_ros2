#ifndef TRT_H
#define TRT_H
#include <NvInfer.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
using namespace nvinfer1;
// 函数声明
bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir);
void prepare_buffers(nvinfer1::ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
void infer(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);
void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
void deserialize_engine(std::string& engine_name, nvinfer1::IRuntime** runtime, nvinfer1::ICudaEngine** engine, nvinfer1::IExecutionContext** context);


#endif