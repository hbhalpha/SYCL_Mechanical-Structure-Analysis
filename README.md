# SYCL_Mechanical-Structure-Analysis
这里是一个基于SYCL与Intel oneAPI实现的异构处理问题程序
## 一.项目背景与介绍
### 针对工程力学中常常出现的工程问题，我们有以下问题需要大规模计算处理：
> 1.针对实际问题中非理想化模型，即在理想化模型经常需要施加随机量才能得到仿真，处理类**不确定性量化（Uncertainty Qualifications，UQ）** 问题 
> 
> 2.针对 非齐次线性方程组 $U\mathbf{x}=\mathbf{F}$ 大规模求解与仿真，
> U代表**刚度矩阵**，F代表**所受外力**，X代表**所构建模型中物体中节点的位移**
> 
> 3.对解向量的统计，如方差，平均值，偏度，概率密度分布函数等
> 
> 4求解最佳受力 $\mathbf{F}$ 使节点位移区域均匀，即固体结构的稳定性
> 
> 5对结构界定稳定，不稳定和临界稳定等类别，通过机器学习，对刚度和力的组合进行分类。
### 主要任务
>运用**蒙特卡罗模拟（Monte Carlo Simulation，MCS）** 对刚度矩阵的真实情况进行模拟
>
>在多次仿真模拟下，求解方程 $U\mathbf{x}=\mathbf{F}$ ，获得在不确定环境下模拟得到的平均解
>
>对解向量 $\mathbf{F}$ 进行分析统计，如方差，平均值，偏度，概率密度分布函数等
>基于**梯度下降（Gradient Descent GD）** 最小化目标函数
> $$\sum_{i,j} (x_i - \bar{x})^2$$
> 
> 基于ann监督学习,实现有效拟合该力学特征的向量
## 二.实现与模拟
### 部分函数功能说明
 **gradient** 函数用于计算某个函数的梯度。在这个情况下，我们要计算的是目标函数，其关于每个元素 $\textbf{X}[i]$ 的偏导数为 $2 * \textbf{X}[i] - 2 * {sum_x} / n$ ，其中 ${sum_x}$ 是 $2 * sum_x$  向量所有元素的和。这个函数通过 **parallel_for** 将计算过程分散到多个工作项上，以提高效率。
 
**gradient_descent** 函数则是使用梯度下降法来更新 $\textbf{F}$ 向量。这个函数首先计算了梯度向量 $\textbf{grad}$ ，然后用 $\textbf{grad}$ 向量来按照梯度下降法的规则更新 $\textbf{F}$ 向量。更新规则是：新的 $\textbf{F}[i]$ 等于原来的 $\textbf{F}[i]$ 减去学习率（ $\textbf{lr}$ ）乘以梯度 $\textbf{grad}[i]$ 。这里的学习率是一个超参数，控制了每次更新的步长。

最后，这个函数将更新后的 $\textbf{F}$ 向量从 $\textbf{SYCL}$ 缓冲区 $F_buf$ 复制回主机内存中的 $\textbf{F}$ 向量。

以上两个函数都使用了 $\textbf{SYCL}$ 的并行计算特性，通过并行处理多个元素，提高了计算效率。这些函数也是典型的机器学习中的操作， $\textbf{gradient}$ 用于计算梯度， **gradient_descent** 则用于利用梯度进行优化。


**solveLinearEquation**  是用来求解线性方程的，它的输入包括一个 $\textbf{SYCL}$ 队列，一个二维向量 $\mathbf{A}$ 表示系数矩阵，以及一个一维向量 $\mathbf{B}$ 表示方程右侧的值。

首先，函数将输入的二维向量 $\mathbf{A}$ 转换为一维向量 $\mathbf{A\_1D}$ ，将其按列存储
。
函数调用 $\textbf{oneapi::mkl::lapack::getrf}$ 函数，它会计算出系数矩阵 $\mathbf{A}$ 的 $\textbf{LU}$ 分解，结果会直接写入 $\mathbf{A\_buf}$ 和 $\mathbf{ipiv\_buf}$ 中。

接着，函数调用 $\textbf{oneapi::mkl::lapack::getrs}$ 函数，它会利用前面计算出的 $\textbf{LU}$ 分解来解线性方程，得到的解会直接写入 $\mathbf{B\_buf}$ 中。

最后，函数将解从 $\mathbf{B\_buf}$ 中复制回一个标准向量 $\mathbf{X}$ ，并将其作为函数的返回值。

通过以上步骤，这个函数完成了线性方程的求解，这个过程主要是通过调用 $\textbf{oneMKL}$ 库中的函数来完成的，而且还利用了 $\textbf{SYCL}$ 的并行计算能力，以提高计算效率。
### 利用SYCL优化实现的并行神经网络
代码分析
在代码中，首先定义了一个 Layer 类，该类代表神经网络中的一层。这个类有一个前向传播函数 forward() 和一个反向传播函数 backward()。其中，前向传播函数用于计算该层的输出，反向传播函数则用于根据反向传播的梯度更新层的权重和偏置。

然后，我们定义了一个 NeuralNetwork 类，这个类中包含了两个 Layer 对象：hidden_layer 和 output_layer。该类也有一个前向传播函数 forward() 和一个反向传播函数 backward()，以及一个 train() 函数，用于训练神经网络。

SYCL优化
在原始代码中，反向传播函数 backward() 中的计算过程是串行的。但实际上，这个过程可以被并行化，因为计算每个神经元的梯度并不依赖于其他神经元的梯度。

所以，在优化后的代码中，我们利用 SYCL 提供的并行编程接口，将反向传播函数中的计算过程并行化，提高了计算效率。在 SYCL 中，通过 queue.submit() 函数提交一个计算任务 h.parallel_for() 函数则用于在设备上并行执行某个操作。

数据输入
在训练神经网络时，我们需要输入训练数据。在优化后的代码中，我们实现了一个 read_data() 函数，用于从文件中读取训练数据。该函数会读取文件中的每一行，将每一行的数据转换为一个 double 类型的向量，然后将这些向量存储在一个二维向量中返回。
## 三.结果验收与性能检测
### 经测试，全部结果验收正确
<img width="316" alt="image" src="https://github.com/hbhalpha/SYCL_Mechanical-Structure-Analysis/assets/122025982/83b58e17-0acc-42ec-b45d-7d95c58ce5f3">
效果如图

<img width="233" alt="image" src="https://github.com/hbhalpha/SYCL_Mechanical-Structure-Analysis/assets/122025982/d7de03f6-8971-4747-86ab-dcdb0326076b">
神经网络并行测试如图

### 性能测试

当矩阵规模为 20 * 20 的时候  ，相同的Py程序与SYCL程序对比:

<img width="388" alt="24c34468cd10d98e6a94709d263c5a4" src="https://github.com/hbhalpha/SYCL_Mechanical-Structure-Analysis/assets/122025982/ae18f107-ba1a-4740-9fb1-e891a9acd95b">

SYCL

<img width="272" alt="fc1ba4ec19af105c5e0804670d89f64" src="https://github.com/hbhalpha/SYCL_Mechanical-Structure-Analysis/assets/122025982/90b8fc68-fb72-4cb0-97f1-541837bbb7f8">

PY

当矩阵规模为 50 * 50 的时候  ，相同的Py程序与SYCL程序对比:

<img width="201" alt="fe765212750092b53e191b508c0a495" src="https://github.com/hbhalpha/SYCL_Mechanical-Structure-Analysis/assets/122025982/8e2f635c-493b-4aa4-bc3a-dc3e1fb79c19">

SYCL

<img width="443" alt="c94b7cc62418ebb323803476a84fea7" src="https://github.com/hbhalpha/SYCL_Mechanical-Structure-Analysis/assets/122025982/5307396e-6b76-490e-8d17-e1496a6951e4">

PY

## 四.结论
项目整体较为成功，并有力地彰显了SYCL语言的针对于大规模计算的优越性




