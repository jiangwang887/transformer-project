# Transformer Model Implementation in PyTorch
从零开始实现的通用 Transformer 模型，可用于处理多种自然语言处理（NLP）任务。（本项目在tinyshakespeare数据集上训练用于模仿生成文本）

## ✨ 功能特性

- **从零实现**: 核心 Transformer 组件（如 `MultiHeadAttention`, `PositionalEncoding`, `EncoderBlock`, `DecoderBlock`）均从零开始编写，易于理解和修改。
- **多任务支持**: 架构设计灵活，通过配置文件可轻松切换不同的 NLP 任务：
  - **语言建模 (Language Modeling)**: 类似 GPT 的仅解码器（Decoder-Only）模式，用于文本生成。
  - **序列到序列 (Sequence-to-Sequence)**: 完整的编码器-解码器（Encoder-Decoder）架构，用于机器翻译等任务。
  - **文本分类 (Text Classification)**: （需在模型中添加分类头）数据加载流程已支持。
- **可配置**: 通过 `src/config.py` 文件可以轻松定义和切换不同的模型超参数、数据集和训练设置。
- **高级训练特性**:
  - 支持从检查点**恢复训练**。
  - 集成 **Cosine Annealing 学习率调度器**。
  - 自动将训练和验证指标（Loss, Perplexity, Learning Rate）记录到 `training_metrics.csv`。
- **文本生成**: `eval.py` 脚本支持使用训练好的模型进行文本生成，并集成了**温度（Temperature）**和 **Top-K 采样**策略。
- **结果可视化**: `results/plot_metrics.py` 脚本可以一键将训练过程中的指标绘制成图表。

## 📂 项目结构

```
.
├── data/                     # (建议) 存放本地数据集
│   └── tinyshakespeare.txt
├── results/                  # 存放结果和图表
│   ├── plot_metrics.py       # 绘图脚本
│   └── metrics_iwslt2017.csv #训练损失
├── src/                      # 核心源代码
│   ├── config.py             # 任务和模型配置文件
│   ├── dataset_utils.py      # 数据加载和预处理
│   └── model.py              # Transformer 模型定义
├── train.py                  # 主训练脚本
├── eval.py                   # 评估和文本生成脚本
├── best_model_checkpoint.pt  # 训练中保存的最佳模型
├── training_metrics.csv      # 训练指标日志
└── requirements.txt          # 项目依赖
```
### 1. 训练模型

使用 `train.py` 脚本开始训练。你可以通过 `--config` 参数选择一个预定义的配置（例如 `iwlst2017`）。

**开始一次新训练:**
```bash
# 在第一个可用 GPU 上训练
python train.py --config iwlst2017
```

**指定 GPU 并设置总轮次:**
```bash
# 在 1 号 GPU 上训练 10 轮
CUDA_VISIBLE_DEVICES=1 python train.py --config tiny_shakespeare --epochs 10
```

### 3. 评估与文本生成

训练完成后，`best_model_checkpoint.pt` 文件会被保存在项目根目录。运行 `eval.py` 来加载模型并。
```bash
python eval.py
```

### 4. 可视化训练过程

运行 `plot_metrics.py` 脚本来将 `training_metrics.csv` 中的数据绘制成图表。
```bash
python results/plot_metrics.py
```
生成的 `training_curves.png` 图像将被保存在 `results/` 目录下。

## 🔧 预置配置

项目中预置了以下配置，可在 `src/config.py` 中查看和修改：

- `tiny_shakespeare`: 用于字符级语言建模。
- `iwslt2017`: 用于英德机器翻译。
- `ag_news`: 用于新闻文本分类。
```


