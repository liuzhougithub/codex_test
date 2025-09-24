数据简介
对某型防暴装甲车传动箱预置了早期故障，利用2个方向垂直的振动加速度传感器采集传动箱振动信号数据、转速脉冲传感器采集输出轴转速数据，其他设备包括计算机、数据采集板以及数据采集软件VibraQuest Pro，采样频率为20.48kHz。一共有10种健康状态，其中包括9种早期混合故障状态和正常状态。在不同类型早期故障状况下，采集传动箱输出轴转速、两个方向振动加速度时序数据，利用训练集进行分析建模、提取故障特征，对测试集数据计算预测装备故障。
数据说明
以齿轮断齿和轴承滚动体故障数据集为例，对数据集情况进行说明，见下表：

文件	文件名	文件说明
齿轮断齿和轴承滚动体故障	break_roller_fz0.1_zs1200.xlsx	共3列，第一列为转速脉冲信号，第二列为垂直方向振动信号，第三列为水平方向振动信号，每列共1015808个数据。

提交示例
初复赛阶段txt文件提交示例
文件内容格式：内容为 txt 格式，包含预测结果的列表，每个结果项包括测试集名称、测试集故障类型2个字段。示例如下：

测试集名称	故障类型
test0001	roller_broken

复现阶段算法源程序文件提交示例
（1）模型所需库文件均放于开头部分，框架统一采用pytorch框架

*代码框架示例：
# 导入所需库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
……
（2）数据集构建流程说明
①设定数据参数
通过滑动窗口划分样本，设置滑动窗口的长度（如500点）和滑动步长（如500点），不允许有重叠。
定义标签映射关系：根据不同故障类型（例如磨损-外圈、断齿-内圈等）来确定类别编号。
②遍历故障数据文件
逐个读取每一个故障类别的文件内容，并转换为矩阵/数组形式，假设每个文件中包含三列轴承信号数据。
③滑动窗口切分样本
使用设定的窗口长度与步长，在时间序列数据上滑动。
每次截取一个固定大小的窗口（例如500×3的矩阵，即500行、3列信号）。
如果窗口数据完整，则作为一个样本保存，同时记录其对应的类别标签。
④数据集整理与保存
将所有样本堆叠成一个三维数组，形状为(样本数,窗口长度,特征数)。
将标签整理为一个一维数组，与样本一一对应。
最终将样本和标签保存为合适的文件（如.npz格式等），方便后续模型训练和实验使用。

（3）数据处理规范
①数据读取
数据shape统一为(num_samples, seq_len, num_channels)。
②批量加载
使用TensorDataset封装样本和标签。
使用DataLoader，训练集shuffle=True，测试集shuffle=False。

*代码框架示例：
# 数据加载
data = np.load(data_path)
samples = torch.tensor(data['samples'], dtype=torch.float32)
labels = torch.tensor(data['labels'], dtype=torch.long)

# 数据集划分
train_size = int(0.7 * len(samples))
test_size = len(samples) - train_size
train_dataset, test_dataset = random_split(TensorDataset(samples, labels), [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
（4）模型构造规范
①类结构
每个模型必须封装为nn.Module子类，命名规则ModelName(nn.Module)。
模型必须包含：init（定义层）、forward（前向传播）。
②参数计算
全连接层输入维度必须通过计算公式得到，避免硬编码。
模型类中加入self.feature_length=…。

*代码框架示例：
class CNN(nn.Module):
def __init__(self, input_channels, num_classes):
super(CNN1D, self).__init__()
# 定义卷积层、池化层、全连接层
...
self.fc1 = nn.Linear(self.feature_length, 128)
self.fc2 = nn.Linear(128, num_classes)

def forward(self, x):
# 前向传播
...
return out
（5）训练流程规范
①模型训练规范
输入：模型、训练集train_loader、损失函数、优化器。
步骤：
设置模型train()；
每个epoch循环数据：前向传播→计算损失→反向传播→更新参数；
输出训练过程的平均loss。
②可以选择使用GPU训练，并在代码中用注释说明。

*代码框架示例：
#自动检测CUDA是否可用并输出使用设备类型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用的设备:", device)
#模型训练
def train_model(model, train_loader, criterion, optimizer, device, epochs=20):
for epoch in range(epochs):
model.train()
for x, y in train_loader:
# 前向传播 + 反向传播 + 更新
print(f"Epoch {epoch+1}/{epochs}, Loss={loss:.4f}")
（6）模型评估规范
①评估指标
统一使用预测准确率和宏平均F1分数。
②可视化展示
使用seaborn.heatmap绘制混淆矩阵，额外可增加tsne图。
③测试流程
使用torch.no_grad()禁止梯度计算。
保存真实标签all_labels和预测结果all_preds。

*代码框架示例：
def evaluate_model(model, loader, device):
model.eval()
all_labels, all_preds = [], []
…
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

accuracy = (all_labels == all_preds).mean() * 100
f1_macro = f1_score(all_labels, all_preds, average="macro")

return accuracy, f1_macro, all_labels, all_preds
（7）实验控制与输出
输入：实验重复训练3次
步骤：
每次新建模型并训练；
优化器与损失函数需要在代码中明确标出，例如使用Adam(lr=0.001)和CrossEntropyLoss；
记录每次评估指标并输出每次结果；
计算并输出最终均值。

*代码框架示例：
for run in range(1, 4):
print(f"\n==== 第 {run} 次训练 ====")
model = CNN(xx,xx,xx).to(device)
#使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
#使用Adam优化器(lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)
……
print(f"第{run}次评估结果:Accuracy={acc:.2f}%, F1-macro={f1_macro:.4f}")

#混淆矩阵（可选，每次画一次）
conf_matrix = confusion_matrix(test_labels, test_preds)
……
#tsne可视化
Tsne=plot_avg_tsne(features_runs, labels_runs, class_names)
……
# ========================
# 输出均值
# ========================
print("\n==== 三次训练结果平均值 ====")
for i in range(3):
print(f"Run {i+1}: Accuracy={acc_list[i]:.2f}%, F1-macro={f1_macro_list[i]:.4f}")

print(f"\n平均 Accuracy = {np.mean(acc_list):.2f}%")
print(f"平均 F1-macro = {np.mean(f1_macro_list):.4f}")

复现阶段程序说明模版
1.程序加载方式
2.程序的主要文件作用说明
3.数据预处理说明（是否对原始数据做了去噪、归一化、特征提取等处理）
4.算法/模型概述
对模型进行详细介绍，并给出架构参数表（如下表）,包括训练轮数及学习率大小。
例如：ACCN模型是由4个卷积层，1个注意力层和2个胶囊层组成。将数据组织成形状为（样本数，6，1000）的张量，并送入一维卷积网络，其中1000表示每个样本的时间序列长度，6表示通道数。为了充分提取信息，卷积层卷积核的大小设计为从大到小。ECA机制的加入并不会改变原有的特征形状。卷积层使用LeakyReLU作为激活函数，胶囊层使用Squash函数作为激活函数。网络模型的构建、训练与测试均在pytorch深度学习框架下基于Python编程语言环境完成，训练时批量大小为16，在训练数据集上迭代更新100次，学习率5×10-4，优化器为Adam，损失函数为边距损失。胶囊层的动态路由算法的迭代次数为3次。
表3 ACCN模型参数设置
层类型 核尺寸 步长 核个数 输出尺寸
输入                   (6,1000)
卷积层（bn+LeakyReLU） 100 4 16 (16,226)
卷积层（bn+LeakyReLU） 20 2 32 (32,104)
卷积层（bn+LeakyReLU） 10 2 64 (64,48)
卷积层（bn+LeakyReLU） 5 2 64 (64,22)
ECA层                  (64,22)
初级胶囊层 4 2 20 (200,16)
数字胶囊层              (7,32)

5.模型先进性及创新性说明
