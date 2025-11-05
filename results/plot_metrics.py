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

    # 4. 创建一个包含多个子图的图形
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
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

    # 调整布局并保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_path)
    print(f"✅ 训练曲线图已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 training_metrics.csv 绘制训练曲线。")
    parser.add_argument("--csv_path", type=str, default="training_metrics.csv",
                        help="训练指标 CSV 文件的路径。")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="保存生成图像的目录。")
    args = parser.parse_args()

    plot_training_curves(args.csv_path, args.output_dir)