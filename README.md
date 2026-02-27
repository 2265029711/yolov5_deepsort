# YOLOv5 + DeepSORT 多目标跟踪系统

基于 YOLOv5 和 DeepSORT 的实时多目标检测与跟踪系统，提供友好的 PyQt5 GUI 界面。

## 功能特点

- 🎯 **实时目标检测**：基于 YOLOv5 的高精度目标检测
- 🔍 **多目标跟踪**：使用 DeepSORT 算法实现稳定的身份跟踪
- 🖥️ **图形化界面**：基于 PyQt5 的现代化 GUI
- 📊 **实时参数显示**：FPS、检测数量、跟踪目标数等实时统计
- ⚙️ **灵活配置**：支持模型选择、置信度阈值调整等参数配置
- 🎬 **视频进度控制**：支持暂停、继续、进度拖动等功能

## 支持的检测类别

- 行人 (person)
- 车辆 (car)

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- PyQt5
- OpenCV
- NumPy

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/yolov5_deepsort.git
cd yolov5_deepsort
```

### 2. 创建虚拟环境

```bash
conda create -n yolov5_deepsort python=3.10
conda activate yolov5_deepsort
```

### 3. 安装依赖

```bash
# 安装 PyTorch (根据你的 CUDA 版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 4. 下载模型权重

将 YOLOv5 模型权重文件 (`.pt`) 放入 `data/weights/` 目录：

```bash
data/weights/
├── yolov5nu.pt
├── yolov5su.pt
└── mars-small128.pb  # DeepSORT 特征提取模型（可选）
```

## 使用方法

### 启动程序

```bash
python main.py
```

### 操作步骤

1. 点击 **选择视频** 加载视频文件
2. 在 **模型参数** 区域选择检测模型和置信度阈值
3. 在 **跟踪对象** 区域勾选需要跟踪的目标类别
4. 点击 **开始** 按钮开始检测跟踪
5. 使用 **暂停** / **继续** 控制播放
6. 点击 **停止** 结束检测

## 项目结构

```
yolov5_deepsort/
├── main.py                 # 程序入口
├── requirements.txt        # 依赖列表
├── README.md              # 项目说明
├── .gitignore             # Git 忽略配置
├── data/
│   ├── weights/           # 模型权重
│   └── videos/            # 测试视频
├── gui/
│   ├── __init__.py
│   ├── main_window.py     # 主窗口
│   ├── control_panel.py   # 控制面板
│   └── video_widget.py    # 视频显示组件
├── utils/
│   ├── __init__.py
│   ├── detector.py        # YOLOv5 检测器封装
│   └── tracker.py         # DeepSORT 跟踪器封装
├── deep_sort/             # DeepSORT 算法实现
└── yolov5/                # YOLOv5 模型代码
```

## 模型选择建议

| 模型 | 速度 | 精度 | 适用场景 |
|------|------|------|----------|
| YOLOv5n | 最快 | 较低 | 实时性要求高 |
| YOLOv5s | 快 | 中等 | 平衡场景 |
| YOLOv5m | 中等 | 较高 | 精度要求较高 |
| YOLOv5l | 慢 | 最高 | 离线高精度分析 |

## 注意事项

- 首次运行会自动下载 YOLOv5 模型（如果本地不存在）
- DeepSORT 的 TensorFlow 特征提取模型为可选组件，缺失时使用备用特征提取方法
- 建议使用 GPU 以获得更好的实时性能

## 技术栈

- [YOLOv5](https://github.com/ultralytics/yolov5) - 目标检测
- [DeepSORT](https://github.com/nwojke/deep_sort) - 多目标跟踪
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI 框架
- [OpenCV](https://opencv.org/) - 图像处理
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 许可证

本项目仅供学习和研究使用。

## 致谢

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [DeepSORT](https://github.com/nwojke/deep_sort)
