import os
import math
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
# ---> 修改 1/4: 导入学习率调度器 <---
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import Transformer # 你自己实现的 Transformer
from src.config import get_config
from src.dataset_utils import get_dataloaders

def generate_causal_mask(size, device):
    """为 decoder 生成因果掩码"""
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

def train_one_epoch(model, dataloader, criterion, optimizer, config, epoch):
    """训练一个 epoch 的核心函数。"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Training]")
    for batch in pbar:
        optimizer.zero_grad()
        device = torch.device(config['device'])
        
        if config['task_type'] == 'language_modeling':
            inputs, targets = batch
            inputs, targets = inputs.to(device).t(), targets.to(device).t()
            causal_mask = generate_causal_mask(inputs.size(0), device)
            output = model(src=inputs, tgt=inputs, src_mask=None, tgt_mask=causal_mask)
            loss = criterion(output.view(-1, config['vocab_size']), targets.reshape(-1))

        elif config['task_type'] == 'sequence_to_sequence':
            src = batch['input_ids'].to(device).t()
            tgt = batch['labels'].to(device).t()
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]
            tgt_mask = generate_causal_mask(tgt_input.size(0), device)
            output = model(src=src, tgt=tgt_input, src_mask=None, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, config['vocab_size']), tgt_out.reshape(-1))
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, config):
    """在验证集上评估模型。"""
    model.eval()
    total_loss = 0
    device = torch.device(config['device'])
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Validation]")
        for batch in pbar:
            if config['task_type'] == 'language_modeling':
                inputs, targets = batch
                inputs, targets = inputs.to(device).t(), targets.to(device).t()
                causal_mask = generate_causal_mask(inputs.size(0), device)
                output = model(src=inputs, tgt=inputs, src_mask=None, tgt_mask=causal_mask)
                loss = criterion(output.view(-1, config['vocab_size']), targets.reshape(-1))
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
    return total_loss / len(dataloader)


def main(args):
    """主函数，负责 orchestrate 整个训练流程"""
    
    # 1. 获取配置并应用命令行覆盖
    config = get_config(args.config)
    if args.epochs is not None:
        config['epochs'] = args.epochs
        print(f"命令行覆盖：将总训练轮次设置为 {config['epochs']}")
    
    if args.local_dataset:
        config['local_dataset_path'] = args.local_dataset

    device = torch.device(config["device"])

    # --- 修正 2: 顺序调整 ---
    # 步骤 2: 首先获取数据加载器。此函数会根据数据动态计算 vocab_size 并更新 config。
    print("--- 正在准备数据加载器 ---")
    train_loader, val_loader, config = get_dataloaders(config)
    print(f"--- 数据准备完成，实际词汇表大小: {config['vocab_size']} ---")

    # 步骤 3: 现在使用更新后的 config (包含正确的 vocab_size) 来初始化模型。
    model = Transformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"],
        task_type=config["task_type"] # <--- 新增: 传递任务类型
    ).to(device)

    # 4. 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=config.get('pad_token_id', -100))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)

    # ---> 修改 2/4: 初始化学习率调度器 <---
    # T_max: 调度器将运行的总轮数
    # eta_min: 学习率的下限
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config.get("min_lr", 1e-5))

    # 5. 初始化训练状态
    start_epoch = 0
    best_val_loss = float('inf')
    best_model_path = os.path.join(os.path.dirname(__file__), "best_model_checkpoint.pt")

    if args.resume and os.path.exists(best_model_path):
        print(f"🔄 从检查点恢复训练: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']

        # ---> 修改 3/4 (部分a): 加载调度器的状态 <---
        if 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            print("学习率调度器状态已加载。")

        print(f"已加载 epoch {start_epoch} 的检查点，将从 epoch {start_epoch + 1} 开始训练。最佳验证损失: {best_val_loss:.4f}")

    print(f"--- 开始在 {device} 上训练 '{args.config}' ---")
    metrics_file = os.path.join(os.path.dirname(__file__), "training_metrics.csv")
    
    if not os.path.exists(metrics_file) or start_epoch == 0:
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,train_ppl,val_ppl,learning_rate\n")

    for epoch in range(start_epoch, config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config, epoch)
        
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, config)
        else:
            val_loss = train_loss 
            if epoch == 0:
                print("[警告] 未找到验证数据加载器，将基于训练损失保存模型。")

        # ---> 修改 4/4: 在每个 epoch 结束后更新学习率 <---
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        train_ppl = math.exp(min(train_loss, 100.0))
        val_ppl = math.exp(min(val_loss, 100.0))

        print(f"Epoch {epoch+1}/{config['epochs']}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}, lr={current_lr:.6f}")

        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{train_ppl:.6f},{val_ppl:.6f},{current_lr:.8f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"🎉 新的最佳模型! 验证损失降低至: {val_loss:.4f}。模型已保存至 -> {best_model_path}")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                # ---> 修改 3/4 (部分b): 保存调度器的状态 <---
                'scheduler_state': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, best_model_path)

    print("--- 训练完成 ---")
    print(f"最终最佳模型保存在: {best_model_path}，其验证损失为: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通用 Transformer 训练脚本")
    parser.add_argument("--config", type=str, default="tiny_shakespeare",
                        help="要使用的配置名称 (例如: tiny_shakespeare, iwslt2017)")
    parser.add_argument("--local_dataset", type=str, default=None,
                        help="可选：本地数据文件路径（优先使用）")
    parser.add_argument("--resume", action="store_true",
                        help="从上一个检查点恢复训练")
    # ---> 新增代码: 添加 --epochs 参数 <---
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖配置中的训练轮次总数")
    args = parser.parse_args()
    
    # 自动检测 tiny_shakespeare 的默认路径
    if args.config == "tiny_shakespeare" and args.local_dataset is None:
        default_path = "./data/tinyshakespeare.txt"
        if os.path.exists(default_path):
             args.local_dataset = default_path
             print(f"使用默认本地数据集路径: {args.local_dataset}")

    main(args)