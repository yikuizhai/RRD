# RRD

RRD 是一个基于 PyTorch 的工业图像异常检测项目，采用教师编码器、特征瓶颈和反向解码器结构，通过多尺度特征差异完成图像级异常检测与像素级异常定位。

当前代码支持多类别联合训练，并支持 MVTec AD、VisA 和自定义 HD 数据集。训练阶段仅使用正常样本，同时通过潜在特征重组生成扰动特征；测试阶段根据教师与解码器特征之间的余弦距离生成异常图。

## 主要功能

- 多类别正常样本联合训练
- Wide ResNet-50-2 教师编码器与反向解码器
- 潜在特征重组增强
- 图像级和像素级异常检测评估
- 输出异常热力图及其与原图的叠加结果
- 支持以下评价指标：
  - 图像级：AUROC、AP、F1
  - 像素级：AUROC、AP、F1、AUPRO

## 项目结构

```text
RRD/
├── multi_train.py    # 多类别训练入口
├── multi_test.py     # 测试、评估与热力图生成入口
├── dataset.py        # 数据预处理及数据集定义
├── resnet.py         # 教师编码器和特征瓶颈
├── de_resnet.py      # 反向 ResNet 解码器
├── metrics.py        # 异常检测评价指标
├── Loss.py           # 损失函数
├── CLFAM.py          # 特征注意力模块
├── HPCAM.py          # 特征注意力模块
├── checkpoints/      # 模型权重输出目录
└── visualizations/   # 异常热力图输出目录
```

## 环境安装

推荐使用带 CUDA 支持的 PyTorch 环境。主要依赖如下：

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Pillow
- OpenCV
- scikit-learn
- scikit-image
- THOP

示例安装命令：

```bash
pip install torch torchvision numpy pillow opencv-python scikit-learn scikit-image thop
```

请根据本机 CUDA 版本，从 [PyTorch 官方网站](https://pytorch.org/get-started/locally/) 选择合适的 PyTorch 安装命令。

首次运行时，预训练的 Wide ResNet-50-2 权重可能会由 PyTorch 自动下载，因此需要网络连接；也可以提前将权重放入 PyTorch 的模型缓存目录。

## 数据集准备

代码要求数据根目录下按类别组织数据。以 MVTec AD 中的 `bottle` 类别为例：

```text
<data_root>/
└── bottle/
    ├── train/
    │   └── good/
    │       ├── 000.png
    │       └── ...
    ├── test/
    │   ├── good/
    │   ├── broken_large/
    │   └── ...
    └── ground_truth/
        ├── broken_large/
        └── ...
```

支持的数据集名称如下：

| `dataset_name` | 数据集类 | 图像格式 |
| --- | --- | --- |
| `mvtec` | `MVTecDataset` | PNG |
| `visa` | `VisaDataset` | JPG（扩展名需为 `.JPG`） |
| `hd` | `HDDataset` | PNG |

异常测试图像需要在 `ground_truth` 中具有对应的掩码。正常测试图像不需要掩码，代码会自动创建全零掩码。

## 训练

当前训练脚本使用代码内配置。运行前请编辑 `multi_train.py` 底部的类别列表和训练参数：

```python
item_list = [
    'carpet', 'bottle', 'hazelnut', 'leather', 'cable',
    'capsule', 'grid', 'pill', 'transistor', 'metal_nut',
    'screw', 'toothbrush', 'zipper', 'tile', 'wood'
]

ckpt = train_multi(
    classes=item_list,
    dataset_name='mvtec',
    data_root='/path/to/mvtec_AD',
    ckpt_path='./checkpoints/rrd_wres50_mvtec.pth',
    image_size=256,
    batch_size=48,
    epochs=250,
    lr=5e-3,
    num_workers=4,
)
```

然后运行：

```bash
python multi_train.py
```

训练过程每 5 个 epoch 在测试集上评估一次。模型优先按照平均图像级 AUROC 保存；当图像级 AUROC 相同时，使用平均像素级 AUROC 作为判断依据。checkpoint 包含特征瓶颈、解码器、类别列表、数据集名称和图像尺寸等信息。

> 注意：训练数据加载器使用 `drop_last=True`。正常训练样本总数应不少于 `batch_size`，否则一个 epoch 内可能没有可用批次。

## 测试与可视化

运行前请编辑 `multi_test.py` 底部配置，使类别、数据集、数据路径和模型权重与训练阶段一致：

```python
evaluate_from_checkpoint(
    classes=item_list,
    dataset_name='mvtec',
    data_root='/path/to/mvtec_AD',
    ckpt_path='./checkpoints/rrd_wres50_mvtec.pth',
    image_size=256,
    num_workers=4,
    save_heatmaps=True,
    vis_root='visualizations',
)
```

然后运行：

```bash
python multi_test.py
```

程序会输出各类别及所有类别平均的图像级、像素级评价指标。启用 `save_heatmaps=True` 后，可视化结果将保存到：

```text
visualizations/<checkpoint名称>/<类别名称>/
```

每张可视化图从左到右依次为原图、异常热力图叠加图，以及可用时的真实掩码。

## 模型配置说明

训练和测试默认均使用：

```python
encoder, bn = wide_resnet50_2(pretrained=True)
decoder = de_wide_resnet50_2(pretrained=False)
```

如果切换为其他 ResNet 主干，需要同时修改训练脚本和测试脚本中的编码器、特征瓶颈与解码器配置，并重新训练对应权重。

程序会自动选择运行设备：CUDA 可用时使用 GPU，否则使用 CPU。实际训练建议使用 GPU。

## License

本项目采用 [MIT License](LICENSE)。
