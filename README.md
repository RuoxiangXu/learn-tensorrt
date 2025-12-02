# NVIDIA TensorRT Study Notes

<div align="center">

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

> Notes collected during my internship at ZTE (June–August 2024)

## Resources
### Official Documentation
Getting Started Guide: [NVIDIA TENSORRT DOCUMENTATION](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
Github: https://github.com/NVIDIA/TensorRT
### Blogs
Basic Introduction and Simple Examples: [TensorRT Detailed Getting Started Guide](https://blog.csdn.net/qq_39523365/article/details/126124010)
Detailed Theory and Process Examples: [All You Need: High-Performance Inference Engine Theory and Practice (TensorRT)](https://developer.aliyun.com/article/995926)
## Basic Concepts
> NVIDIA® TensorRT™ is an SDK for optimizing trained deep learning models to enable high-performance inference. TensorRT contains a deep learning inference optimizer for trained deep learning models, and a runtime for execution. After you have trained your deep learning model in a framework of your choice, TensorRT enables you to run it with higher throughput and lower latency.

TensorRT is a C++ inference framework that can run on various NVIDIA GPU hardware platforms. Models trained using Pytorch, TensorFlow, or other frameworks can be converted to TensorRT format, and then the TensorRT inference engine can be used to run the model, thereby improving the model's running speed on NVIDIA GPUs. The speed improvement is quite significant.
## How to Speed Up
- **Operator Fusion (Layer and Tensor Fusion)**: Simply put, by fusing some computational ops or removing redundant ops to reduce data flow and frequent use of video memory for acceleration
- **Quantization**: Quantization refers to using INT8 or FP16 and TF32, which are different from the conventional FP32 precision. These precisions can significantly improve model execution speed without affecting the original model's accuracy
- **Kernel Auto-Tuning**: Based on different GPU architectures, number of SMs, kernel frequency, etc. (e.g., 1080Ti and 2080Ti), select different optimization strategies and computation methods to find the most suitable computation method for the current architecture
- **Dynamic Tensor Memory**: As we all know, allocating and releasing video memory is time-consuming. By adjusting some strategies, we can reduce the number of these operations in the model, thereby reducing model running time
- **Multi-Stream Execution**: Using CUDA's stream technology to maximize parallel operations

## Deployment
TensorRT can be deployed in three ways officially provided:
- Integrated into TensorFlow, such as TF-TRT, which is relatively convenient to operate but the acceleration effect is not very good;
- Running models in TensorRT Runtime environment, which is using TensorRT directly;
- Used with serving frameworks, the best match is the official triton-server, which perfectly supports TensorRT and is excellent for production environments!

Reference Documentation:
[TensorRT to Triton](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/deploy_to_triton/README.md); [Pytorch to ONNX, TensorRT Pitfall Record](https://zhuanlan.zhihu.com/p/581237530)
## Precision Types
Supports FP32, FP16, INT8, TF32, etc. These types are commonly used.
- **FP32**: Single-precision floating-point, nothing much to say, the most common data format in deep learning, used in both training and inference;
- **FP16**: Half-precision floating-point, uses half the memory compared to FP32, has corresponding instruction sets, much faster than FP32;
- **TF32**: A data type supported by the third-generation Tensor Core, a truncated Float32 data format that reduces the 23 mantissa bits in FP32 to 10 bits, while the exponent bits remain 8 bits, with a total length of 19 (=1+8+10). It maintains the same precision as FP16 (both have 10 mantissa bits), while also maintaining the dynamic range of FP32 (both have 8 exponent bits);
- **INT8**: Integer type, uses half the memory compared to FP16, has corresponding instruction sets, can be used for acceleration after model quantization.

Note: FP16 is not necessarily faster than FP32. This depends on the number of different precision computing units in the device. For example, on GeForce 1080Ti devices, FP16 computing units are far fewer than FP32, so after pruning, efficiency may actually decrease, while GeForce 2080Ti is the opposite.
## ONNX to TensorRT
Assuming TensorRT is already installed:
TODO
