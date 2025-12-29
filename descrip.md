任务说明书与详细设计报告：GS-SSC (v2.0)
1. 项目概述
项目名称：基于大模型先验与高斯体细化的自监督语义场景补全 (Self-Supervised Semantic Scene Completion with Large-Model Prior and Gaussian Refinement)

核心目标：仅使用单目图像序列（无需 3D Ground Truth），利用 2D 图像和语义伪标签作为监督信号，实现 3D 场景的占据预测 (Occupancy) 和语义补全 (Semantic Completion) (
核心创新点：
1. 几何初始化：利用 MapAnything 替代随机初始化，提供高质量的几何先验，解决 SSC 任务中“无中生有”的冷启动难题 
2. 强语义表征：利用 Frozen DINOv2 提取强泛化性的 2D 特征，并通过 Image Cross-Attention 注入 3D 高斯体，提升对弱纹理区域的理解
3. 高效渲染与监督：使用 gsplat 库替代传统的 NeRF 渲染，支持 RGB 和语义的多通道快速光栅化，配合 S4C 的自监督策略进行训练
4. 补全机制：复用 GaussianFormer 的架构（Self-Attn/SparseConv），使离散的高斯点具备感知上下文并向空洞区域“生长”的能力 (4)。

---
2. 系统架构与数据流 (Pipeline)
系统分为四个阶段：输入与特征提取、几何初始化、场景细化演进、渲染与自监督。
Phase 1: 输入与预处理 (Input & Pre-processing)
目标：准备几何初猜和 2D 语义特征。
- 数据集与采样 (参照 S4C) 
- 数据源：KITTI-360；存储目录： /data/lmh_data/KITTI360
- 采样策略：单视角输入，采样帧进行监督
- 伪标签： 存储目录：/data/lmh_data/KITTI360/panoptic_deeplab_R101_os32_cityscapes_hr
- 2D 特征提取 (Feature Extraction)：
  - 模型：DINOv2 (ViT-Base/Large)。
  - 状态：Frozen (冻结参数)，不进行梯度更新，以保留大模型的泛化能力。
  - 适配器 (Adapter)：由于 DINOv2 输出维度与后续模块可能不匹配，且为单尺度，需通过一个轻量级 Conv2d 模块将特征维度映射（如映射至 256 维），并可能通过上/下采样构建伪多尺度特征供 ICA 使用。
  - 输出：特征图 $F_{dino}$。
Phase 2: 基于先验的初始化 (Initialization)
目标：从 2D 图像获得 3D 场景的初始点云。
- MapAnything实现了与gsplat的集成，直接参考MapAnything由输入图片得到初始化高斯
- Thinking：是否离线 (Offline) 进行以节省训练显存
  - 是否将具体实现分为得到点云和高斯体实例化两步
    - 高斯体实例化：
      - 将 $P_{init}$ 转换为 3D Gaussians。位置 $\mu$ 继承点云坐标，其余属性按标准初始化。
Phase 3: 场景细化与补全 (Refinement Loop)
目标：利用 GaussianFormer 架构，根据图像特征优化高斯属性并填补几何空洞。
- 
- 高斯属性定义 (参考 GaussianFormer) 8：
- 为了同时满足“补全演化”和“S4C 渲染监督”，每个高斯点 $$g_i$$ 包含以下属性：
  1. 几何属性：位置 $\mu \in \mathbb{R}^3$，旋转 $q \in \mathbb{R}^4$，缩放 $s \in \mathbb{R}^3$，不透明度 $\alpha \in \mathbb{R}$。
  2. 演化特征 (Anchor Feature)：$f \in \mathbb{R}^C$。这是 GaussianFormer 的核心，用于 Transformer 交互和回归残差。
  3. 渲染属性 (用于 S4C 监督)：
    - 颜色：球谐系数 (SH) 或 RGB，用于光度损失。
    - 语义：Semantic Logits $\in \mathbb{R}^{Classes}$，用于语义损失。
- 
- 演化模块 (Refinement Network) 参考GaussianFormer实现
1. Self-Encoding (3D Interaction)：
  - 高斯体素化 -> Sparse 3D CNN (MinkowskiEngine/SpConv) -> 去体素化。
  - 目的：让高斯感知周围环境，学习几何连续性，填充空隙。
2. Image Cross-Attention (ICA)：
  - Query: 高斯特征 $f$ + PosEnc。
  - Key/Value: DINOv2 特征图 $F_{dino}$。
  - 目的：将 DINOv2 强大的语义理解注入到 3D 高斯中。
3. Refinement Head:
  - 预测属性残差 $\Delta \mu, \Delta s, \Delta RGB, \Delta Semantics$。
  - Densification: 基于梯度和 Opacity 对高斯进行分裂或克隆，实现物理上的“补全”。
Phase 4: 渲染与自监督 (Rendering & Supervision)
目标：在无 3D GT 的情况下训练网络。
- 渲染引擎：gsplat。
  - 利用 gsplat 的灵活性，执行两次光栅化（或多通道一次性光栅化）：
  - Pass 1 (RGB): 渲染 $H \times W \times 3$ 的彩色图 $\hat{I}_{rgb}$。
  - Pass 2 (Semantic): 渲染 $H \times W \times C$ 的语义特征图 $\hat{I}_{sem}$ (10)。
- 损失函数 (参照 S4C) (11)(11)(11)(11)：
1. $L_{photo}$: $|\hat{I}_{rgb} - I_{input}|$ (L1 + SSIM)。
2. $L_{sem}$: CrossEntropy($\hat{I}_{sem}$, $S_{pseudo}$). $S_{pseudo}$ 来自 Mask2Former。
3. 平滑函数

---
3. 实现指南与环境配置报告
为了确保 S4C (旧 PyTorch)、MapAnything (新 PyTorch)、GaussianFormer (MMDet3D) 和 gsplat (新 CUDA) 之间的兼容性，我们必须构建一个现代化的统一环境。
3.1 兼容性分析与环境推荐
- S4C: 原代码基于 NeRF，对版本要求不高，逻辑可移植。
- MapAnything: 依赖较新的 Transformer 库。
- gsplat: 强烈建议 PyTorch 2.0+ 和 CUDA 11.8+ 以获得最佳性能。
- GaussianFormer: 依赖 mmdet3d。这是最容易冲突的部分。建议不安装完整的 mmdet3d，而是将其核心的 SparseConv 和 Transformer 模块代码剥离出来，或者直接使用 spconv 库实现 3D 卷积。
推荐环境配置 (Dockerfile / Conda)：
Bash
# 1. 基础环境
conda create -n gs-ssc python=3.9
conda activate gs-ssc

# 2. PyTorch (选择支持 CUDA 11.8 或 12.1 的版本)# gsplat 和 xformers 对 PyTorch 版本敏感，推荐 2.1.0
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 核心依赖# MapAnything & DINOv2 需要 transformers 和 xformers
pip install transformers xformers 

# gsplat (高效渲染)
pip install gsplat

# MinkowskiEngine 或 spconv (用于 3D 稀疏卷积 - Phase 3)# 推荐 spconv，安装更简单，且新版支持 PyTorch 2.x
pip install spconv-cu118 

# 图像处理与工具
pip install opencv-python numpy pandas matplotlib plyfile timm
3.2 详细实现步骤
Step 1: 数据预处理 (Offline Preparation)
参考 S4C 的数据处理方式：
1. Images: 提取 RGB 图像序列。
2. Poses: 提取对应的相机位姿 (KITTI 格式转常用格式)。
3. Pseudo-Labels: 运行 Mask2Former，保存每帧的 2D 语义分割结果 (.png 格式, 存 int 类标)。
4. MapAnything Init: 运行 MapAnything 模型，为每个场景保存一个初始点云 (.ply 格式)。
5. DINOv2 Features (Optional): 如果显存不足，提前提取 DINOv2 特征并保存为 .npy。
Step 2: 模型构建
1. GaussianModel: 修改标准 GS 模型，增加 semantic_logits 属性。
2. GeometryRefiner:
  - 实现一个基于 spconv 的 U-Net 结构（替代 GaussianFormer 依赖的 mmdet3d 模块）。
  - 实现 CrossAttention 模块，Query 为 3D 点特征，Key/Value 为 DINOv2 特征。
3. Renderer: 封装 gsplat.rasterization，提供 render_rgb 和 render_semantic 两个接口。
Step 3: 评估 (Evaluation) (12)
采用 GaussianFormer 的评估协议：
1. 训练完成后，冻结高斯体。
2. 体素化 (Voxelization): 设置一个固定分辨率的 Grid (如 0.2m 体素)。
3. 投票 (Voting): 查询每个 Voxel 内部及中心附近的高斯点，取其语义 Logits 的最大值作为该 Voxel 的类别。如果 Voxel 内无高斯，则标记为 Empty。
4. 指标计算: 将预测的 Voxel Grid 与 Ground Truth Occupancy (仅用于评估，不用于训练) 进行对比，计算 mIoU 和 IoU。
3.3 潜在隐患排查
1. DINOv2 维度失配: DINOv2 (Base) 输出 768 维，高斯特征通常 128/256 维。必须实现一个 Learnable Linear Layer 或 Conv Layer 将 DINOv2 降维，否则显存会炸。
2. 坐标系对齐: MapAnything 输出的点云、S4C 提供的 Pose 以及 gsplat 的相机模型必须在同一个坐标系下（通常是 OpenCV 或 OpenGL 坐标系）。这是最容易出错的地方，务必在 Phase 1 可视化初始点云和相机视锥进行验证。
检查这个repository和上面的描述是否完全一致
