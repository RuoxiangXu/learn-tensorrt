# NVIDIA TensorRT 学习笔记

<div align="center">

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

> 记录自本人于 2024.6-2024.8 在中兴通讯实习期间的学习笔记

## 资源
### 官方文档
上手指南：[NVIDIA TENSORRT DOCUMENTATION](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
Github: https://github.com/NVIDIA/TensorRT
### 博客
基本介绍与简单例子：[TensorRT详细入门指南](https://blog.csdn.net/qq_39523365/article/details/126124010)
详细理论与流程举例：[一篇就够：高性能推理引擎理论与实践 (TensorRT)](https://developer.aliyun.com/article/995926)
## 基本概念
> NVIDIA® TensorRT™ is an SDK for optimizing trained deep learning models to enable high-performance inference. TensorRT contains a deep learning inference optimizer for trained deep learning models, and a runtime for execution. After you have trained your deep learning model in a framework of your choice, TensorRT enables you to run it with higher throughput and lower latency.

TensorRT是可以在NVIDIA各种GPU硬件平台下运行的一个C++推理框架。我们利用Pytorch、TF或者其他框架训练好的模型，可以转化为TensorRT的格式，然后利用TensorRT推理引擎去运行我们这个模型，从而提升这个模型在英伟达GPU上运行的速度。速度提升的比例是比较可观的。
## 如何提速
- 算子融合(层与张量融合)：简单来说就是通过融合一些计算op或者去掉一些多余op来减少数据流通次数以及显存的频繁使用来提速
- 量化：量化即IN8量化或者FP16以及TF32等不同于常规FP32精度的使用，这些精度可以显著提升模型执行速度并且不会影响原先模型的精度
- 内核自动调整：根据不同的显卡构架、SM数量、内核频率等（例如1080TI和2080TI），选择不同的优化策略以及计算方式，寻找最合适当前构架的计算方式
- 动态张量显存：我们都知道，显存的开辟和释放是比较耗时的，通过调整一些策略可以减少模型中这些操作的次数，从而可以减少模型运行的时间
- 多流执行：使用CUDA中的stream技术，最大化实现并行操作

## 部署相关
部署TensorRT的方式，官方提供了三种：
- 集成在Tensorflow中使用，比如TF-TRT，这种操作起来比较便捷，但是加速效果并不是很好；
- 在TensorRT Runtime环境中运行模型，就是直接使用TensorRT；
- 搭配服务框架使用，最配的就是官方的triton-server，完美支持TensorRT，用在生产环境杠杠的！

参考文档：
[TensorRT to Triton](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/deploy_to_triton/README.md);[Pytorch转onnx, TensorRT踩坑实录](https://zhuanlan.zhihu.com/p/581237530)
## 权重精度
支持FP32、FP16、INT8、TF32等,这几种类型都比较常用。
- FP32：单精度浮点型，没什么好说的，深度学习中最常见的数据格式，训练推理都会用到；
- FP16：半精度浮点型，相比FP32占用内存减少一半，有相应的指令值，速度比FP32要快很多；
- TF32：第三代Tensor Core支持的一种数据类型，是一种截短的 Float32 数据格式，将FP32中23个尾数位截短为10bits，而指数位仍为8bits，总长度为19(=1+8 +10)。保持了与FP16同样的精度(尾数位都是 10 位），同时还保持了FP32的动态范围指数位都是8位)；
- INT8：整型，相比FP16占用内存减小一半，有相应的指令集，模型量化后可以利用INT8进行加速。

需要注意的是：不一定FP16就一定比FP32的要快。这取决于设备的不同精度计算单元的数量，比如在GeForce 1080Ti设备上由于FP16的计算单元要远少于FP32的，裁剪后反而效率降低，而GeForce 2080Ti则相反。
## ONNX转TensorRT
假设已安装好TensorRT：
TODO
