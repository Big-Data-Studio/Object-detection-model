# Object-detection-model
面向电梯复杂场景的局部特征分块标注电动车检测使用模型

## 维护者
[您的姓名/组织名]

## 许可证
本项目采用 MIT 许可证 

# YOLOv5目标检测项目文档

## 项目概述
本项目基于YOLOv5深度学习框架实现目标检测功能。YOLOv5是一个高效的单阶段目标检测器，具有快速的检测速度和良好的检测精度。

## 环境要求
- Python 3.8+
- PyTorch >= 1.7.0
- CUDA（推荐用于GPU加速）
- 其他依赖项请参考 `requirements.txt`

## 项目结构
```
.
├── yolov5/          # YOLOv5核心代码
└── datasets/        # 数据集目录
```

## 快速开始

### 1. 环境配置
```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
1. 将数据集放置在 `datasets` 目录下
2. 按照YOLOv5格式组织数据：
   - images/：存放图片
   - labels/：存放标注文件
   - train.txt：训练集图片路径列表
   - val.txt：验证集图片路径列表

### 3. 模型训练
```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt
```

### 4. 模型推理
```bash
python detect.py --source path/to/images --weights runs/train/exp/weights/best.pt
```

## 主要功能
1. 目标检测训练
2. 目标检测推理
3. 模型导出（ONNX、TensorRT等）
4. 性能评估

## 性能指标
- 模型大小：具体取决于选择的YOLOv5版本（nano/small/medium/large/xlarge）
- 推理速度：取决于硬件配置和模型大小
- 检测精度：mAP@0.5, mAP@0.5:0.95

## 注意事项
1. 训练前请确保数据集格式正确
2. GPU训练时注意显存占用
3. 建议从小模型开始调试

## 常见问题
1. 如遇到CUDA相关错误，请检查PyTorch版本与CUDA版本是否匹配
2. 数据集格式问题可参考YOLOv5官方文档

## 参考资料
- [YOLOv5官方仓库](https://github.com/ultralytics/yolov5)
- [YOLO论文](https://arxiv.org/abs/2004.10934)

# YOLOv10 目标检测项目技术文档

## 1. 项目概述

本项目基于YOLOv10（You Only Look Once version 10）实现目标检测功能。YOLOv10是一个高效的实时目标检测系统，在保持高精度的同时提供了优秀的推理速度。

### 1.1 主要特点

- 实时目标检测
- 多种模型尺寸选择（从nano到extra large）
- 高精度和快速推理能力
- 支持多种数据格式
- 易于部署和使用

## 2. 系统要求

### 2.1 硬件要求
- CPU: 建议使用多核处理器
- GPU: 建议使用NVIDIA GPU（用于训练和快速推理）
- RAM: 最小8GB，建议16GB或更高
- 存储空间: 至少10GB可用空间

### 2.2 软件要求
- Python 3.8或更高版本
- CUDA Toolkit（如使用GPU）
- 相关Python包（见requirements.txt）

## 3. 安装说明

### 3.1 环境配置
```bash
# 克隆项目
git clone [项目地址]

# 安装依赖
pip install -r requirements.txt
```

### 3.2 模型文件
项目提供多种预训练模型：
- yolov10n.pt (11MB) - 轻量级模型
- yolov10s.pt (31MB) - 小型模型
- yolov10m.pt (64MB) - 中型模型
- yolov10l.pt (100MB) - 大型模型
- yolov10x.pt (122MB) - 超大型模型

## 4. 使用说明

### 4.1 基本使用
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov10n.pt')  # 加载nano模型

# 进行预测
results = model('path/to/image.jpg')
```

### 4.2 API说明
项目提供了app.py作为Web API接口，支持：
- HTTP POST请求进行图像检测
- 实时视频流检测
- 批量图像处理

## 5. 文件结构说明

- `/ultralytics/` - 核心代码目录
- `/tests/` - 测试用例
- `/docs/` - 文档目录
- `/examples/` - 示例代码
- `/figures/` - 图表和示例图片
- `app.py` - Web API实现
- `requirements.txt` - 项目依赖

## 6. 性能指标

不同模型版本的性能对比：
- YOLOv10-nano: 适用于边缘设备，低延迟
- YOLOv10-small: 平衡性能和资源占用
- YOLOv10-medium: 提供更好的检测精度
- YOLOv10-large: 高精度检测
- YOLOv10-xlarge: 最高精度，适用于要求极高精度的场景

# SSD：基于 PyTorch 的单发多框检测器实现

这是一个基于 PyTorch 的 SSD（Single Shot MultiBox Detector）目标检测实现，支持 VGG 和 MobileNetV2 两种骨干网络。

## 快速开始

### 安装
```bash
# 克隆仓库
git clone https://github.com/your-username/ssd-pytorch.git
cd ssd-pytorch

# 安装依赖
pip install -r requirements.txt
```

### 下载预训练权重
- [VGG SSD 权重](https://github.com/bubbliiiing/ssd-pytorch/releases/download/v1.0/ssd_weights.pth) (mAP@0.5: 78.55%)
- [MobileNetV2 SSD 权重](https://github.com/bubbliiiing/ssd-pytorch/releases/download/v1.0/mobilenetv2_ssd_weights.pth) (mAP@0.5: 71.32%)

### 快速推理
```python
from predict import predict_image

# 单张图片预测
predict_image("img/street.jpg")
```

## 在自定义数据集上训练

### 1. 准备数据集
按照 VOC 格式组织数据集：
```
VOCdevkit/
├── VOC2007/
    ├── Annotations/
    │   ├── 000001.xml
    │   ├── 000002.xml
    │   └── ...
    ├── JPEGImages/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── ImageSets/
        └── Main/
            ├── train.txt
            └── val.txt
```

### 2. 配置类别
创建 `model_data/classes.txt` 文件，包含你的类别名称：
```
类别1
类别2
...
```

### 3. 开始训练
```bash
python train.py
```

`train.py` 中的主要训练参数：
- `--backbone`：选择骨干网络，可选 'vgg' 或 'mobilenetv2'
- `--input_shape`：输入图像尺寸（默认：300x300）
- `--batch_size`：训练批次大小
- `--learning_rate`：初始学习率

## 模型架构
SSD 模型包含以下主要组件：
- 骨干网络：VGG16 或 MobileNetV2
- 特征金字塔网络
- 多尺度检测头
- 默认框生成
- 非极大值抑制（NMS）

## 性能指标

| 骨干网络 | 数据集 | 输入尺寸 | mAP@0.5 | FPS (GTX 1080Ti) |
|:--------:|:-------:|:----------:|:-------:|:----------------:|
| VGG16 | VOC07+12 | 300x300 | 78.55% | 82 |
| MobileNetV2 | VOC07+12 | 300x300 | 71.32% | 110 |

## 引用
如果你觉得这个实现对你有帮助，请考虑引用：
```bibtex
@article{liu2016ssd,
  title={SSD: Single Shot MultiBox Detector},
  author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C},
  journal={European Conference on Computer Vision},
  year={2016}
}
```

## 开源协议
本项目采用 [MIT 协议](LICENSE) 开源。

## 致谢
- 原始 SSD 论文：[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- 参考实现：
  - https://github.com/pierluigiferrari/ssd_keras
  - https://github.com/kuhung/SSD_keras 