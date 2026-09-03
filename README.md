# MNIST CNN：PyTorch 与 Jittor 对齐实验

## 1. 项目简介

本项目分别使用 PyTorch 和 Jittor 实现一个简单的卷积神经网络，
完成 MNIST 手写数字分类任务。

实验目的是学习模型训练的完整流程，并验证相同算法在两个深度学习
框架中的实现与性能是否基本一致。

完整流程包括：

- 数据读取与预处理
- CNN 模型搭建
- Loss 计算
- 反向传播
- 模型参数更新
- 测试集评估
- 模型与实验结果保存

## 2. 数据集

实验使用 MNIST in CSV 数据集。

- 训练集：60,000 张图片
- 测试集：10,000 张图片
- 图片尺寸：28 × 28
- 图片通道：1 个灰度通道
- 分类数量：10 类，对应数字 0～9

CSV 文件的第一列为数字标签，剩余 784 列为图片像素。

数据预处理包括：

1. 将 784 个像素恢复为 `1 × 28 × 28` 的图片；
2. 将像素值从 `[0, 255]` 归一化至 `[0, 1]`；
3. 将标签转换为整数类型。

## 3. 模型结构

两个框架采用相同的 SimpleCNN：

| 层 | 输入形状 | 输出形状 | 作用 |
|---|---|---|---|
| Conv1 + ReLU | 1×28×28 | 16×28×28 | 提取初级图像特征 |
| MaxPool | 16×28×28 | 16×14×14 | 将长宽缩小一半 |
| Conv2 + ReLU | 16×14×14 | 32×14×14 | 提取更复杂的特征 |
| MaxPool | 32×14×14 | 32×7×7 | 再次缩小特征图 |
| Flatten | 32×7×7 | 1568 | 将特征图展开 |
| Linear | 1568 | 10 | 输出十个分类分数 |

模型共有 20,490 个可训练参数。

## 4. 训练设置

| 配置 | 数值 |
|---|---|
| Epoch | 3 |
| Batch size | 64 |
| 优化器 | Adam |
| 学习率 | 0.001 |
| Loss | Cross Entropy Loss |
| Jittor 版本 | 1.3.11.0 |
| Jittor 设备 | CPU |

PyTorch 的框架版本和运行设备可在对应 Notebook 中查看。

## 5. PyTorch 与 Jittor 对齐结果

| Framework | Epochs | Batch Size | Learning Rate | Test Loss | Test Accuracy |
|---|---:|---:|---:|---:|---:|
| PyTorch | 3 | 64 | 0.001 | 0.04575 | 98.46% |
| Jittor | 3 | 64 | 0.001 | 0.04974 | 98.22% |

两个框架的测试准确率相差 0.24 个百分点，即在 10,000 张测试图片中
相差 24 张，实验结果基本对齐。

由于本次实验没有固定随机种子，模型初始化和训练数据打乱顺序不同，
因此 Loss 和准确率存在轻微差异是正常现象。

## 6. PyTorch 与 Jittor 的主要代码差异

| 功能 | PyTorch | Jittor |
|---|---|---|
| 张量 | `torch.Tensor` | `jt.Var` |
| 模型前向函数 | `forward()` | `execute()` |
| 反向传播 | `loss.backward()` | `optimizer.step(loss)` |
| 参数更新 | `optimizer.step()` | `optimizer.step(loss)` |
| 模型参数文件 | `.pth` | `.pkl` |

## 7. 项目结构

```text
mnist-pytorch-jittor-alignment/
├── pytorch/
│   ├── pytorch-mnist-cnn.ipynb
│   └── results/
│       ├── mnist_cnn.pth
│       ├── training_history.csv
│       ├── pytorch_test_results.json
│       └── loss_curve.png
│
├── jittor/
│   ├── jittor-mnist-cnn.ipynb
│   └── results/
│       ├── jittor_mnist_cnn.pkl
│       ├── training_history.csv
│       ├── jittor_test_results.json
│       └── loss_curve.png
│
└── README.md
```

## 8. 如何复现实验（供仓库使用者参考）

> 本项目的实验结果已经生成并保存在 `results` 文件夹中，
> 当前无需重新训练。以下步骤仅供希望从头复现实验的使用者参考。

### PyTorch 版本

打开 `pytorch/pytorch-mnist-cnn.ipynb`，按照 Notebook 中的单元格顺序运行。

### Jittor 版本

首先安装 Jittor：

`pip install jittor==1.3.11.0`

在 Kaggle 当前环境中，安装完成后需要重启 Session。重启后跳过安装
单元格，从 Jittor 环境检查单元格开始依次运行。

## 9. 实验结论

本实验使用 Jittor 成功实现了与 PyTorch 结构一致的 CNN，并在 MNIST
测试集上取得 98.22% 的准确率。其结果与 PyTorch 版本的 98.46%
基本一致，验证了模型迁移和训练流程的正确性。
