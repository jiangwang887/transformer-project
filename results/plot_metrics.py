import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_training_curves(csv_path, output_dir="results"):
    """
    从 CSV 文件读取训练指标并绘制训练/验证曲线图。

    Args:
        csv_path (str): 包含训练指标的 CSV 文件路径。
        output_dir (str): 保存图像的目录。
    """
    # 1. 检查 CSV 文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 找不到指标文件 '{csv_path}'。")
        print("请先运行 train.py 来生成该文件。")
        return

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 3. 使用 pandas 读取数据
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"错误: '{csv_path}' 文件为空，无法绘制图形。")
        return
    
    if df.empty:
        print(f"警告: '{csv_path}' 中没有数据，跳过绘图。")
        return

    # ---> 修正 1: 检查是否存在 BLEU 分数，以决定子图数量 <---
    has_bleu = 'bleu_score' in df.columns and df['bleu_score'].notna().any()
    num_plots = 4 if has_bleu else 3
    fig_height = 20 if has_bleu else 15

    # 4. 创建一个包含多个子图的图形
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, fig_height))
    fig.suptitle('Transformer Training Metrics', fontsize=16)

    epochs = df['epoch']

    # --- 子图 1: 损失 (Loss) ---
    axes[0].plot(epochs, df['train_loss'], 'o-', label='Train Loss')
    axes[0].plot(epochs, df['val_loss'], 'o-', label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # --- 子图 2: 困惑度 (Perplexity) ---
    axes[1].plot(epochs, df['train_ppl'], 'o-', label='Train Perplexity')
    axes[1].plot(epochs, df['val_ppl'], 'o-', label='Validation Perplexity')
    axes[1].set_title('Training and Validation Perplexity')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].legend()
    axes[1].grid(True)

    # --- 子图 3: 学习率 (Learning Rate) ---
    axes[2].plot(epochs, df['learning_rate'], 'o-', label='Learning Rate', color='green')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].legend()
    axes[2].grid(True)

    # ---> 修正 2: 如果有 BLEU 分数，则绘制第四个子图 <---
    if has_bleu:
        axes[3].plot(epochs, df['bleu_score'], 'o-', label='BLEU Score', color='purple')
        axes[3].set_title('Validation BLEU Score')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('BLEU')
        axes[3].legend()
        axes[3].grid(True)

    # 调整布局并保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # ---> 修正 3: 根据输入文件名生成更有意义的输出文件名 <---
    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, f"{base_filename}_curves.png")
    plt.savefig(output_path)
    print(f"✅ 训练曲线图已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 training_metrics.csv 绘制训练曲线。")
    # ---> 修正 4: 更改默认值以反映新的文件名格式 <---
    parser.add_argument("--csv_path", type=str, default="results/metrics_iwslt2017.csv",
                        help="训练指标 CSV 文件的路径。")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="保存生成图像的目录。")
    args = parser.parse_args()

    plot_training_curves(args.csv_path, args.output_dir)