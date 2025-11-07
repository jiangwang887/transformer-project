import os
import math
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
# from torch.optim.lr_scheduler import CosineAnnealingLR # <--- 我们不再需要这个
# ---> 新增: 导入 BLEU 计算和翻译函数 <---
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer
# ---> 修正: 导入新的 beam search 函数 <---
from eval import translate_sentence_beam_search

from src.model import Transformer
from src.config import get_config
from src.dataset_utils import get_dataloaders

def generate_causal_mask(size, device):
    """为 decoder 生成因果掩码"""
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

def train_one_epoch(model, dataloader, criterion, optimizer, config, epoch, global_step):
    """训练一个 epoch 的核心函数。"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Training]")
    for batch in pbar:
        # ---> 新增: 学习率预热 <---
        # 我们现在按步数(step)而不是轮次(epoch)来调整学习率
        d_model = config['d_model']
        warmup_steps = config.get('warmup_steps', 4000) # 从config获取，提供默认值
        lr = (d_model ** -0.5) * min((global_step + 1) ** -0.5, (global_step + 1) * warmup_steps ** -1.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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

            # ---> 新增: 创建源句子的 padding mask <---
            # pad_token_id 在 get_dataloaders 中由 tokenizer 设置
            pad_token_id = config['pad_token_id']
            # src 的 shape 是 [seq_len, batch_size], mask 需要 [batch_size, seq_len]
            src_padding_mask = (src == pad_token_id).transpose(0, 1)

            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]
            tgt_mask = generate_causal_mask(tgt_input.size(0), device)
            # ---> 修正: 将 src_padding_mask 传递给模型 <---
            output = model(src=src, tgt=tgt_input, src_mask=src_padding_mask, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, config['vocab_size']), tgt_out.reshape(-1))
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item(), "lr": lr})
        global_step += 1 # <--- 步数加一
        
    return total_loss / len(dataloader), global_step # <--- 返回更新后的步数

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
            
            # ---> 新增: 添加对 sequence_to_sequence 任务的处理逻辑 <---
            elif config['task_type'] == 'sequence_to_sequence':
                src = batch['input_ids'].to(device).t()
                tgt = batch['labels'].to(device).t()

                # ---> 新增: 创建源句子的 padding mask <---
                pad_token_id = config['pad_token_id']
                src_padding_mask = (src == pad_token_id).transpose(0, 1)

                tgt_input = tgt[:-1, :]
                tgt_out = tgt[1:, :]
                tgt_mask = generate_causal_mask(tgt_input.size(0), device)
                # ---> 修正: 将 src_padding_mask 传递给模型 <---
                output = model(src=src, tgt=tgt_input, src_mask=src_padding_mask, tgt_mask=tgt_mask)
                loss = criterion(output.view(-1, config['vocab_size']), tgt_out.reshape(-1))

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
    return total_loss / len(dataloader)


# ---> 新增: 计算 BLEU 分数的函数 <---
def calculate_bleu(model, tokenizer, val_loader, config, num_examples=20):
    """在验证集上计算 BLEU 分数"""
    model.eval()
    references = []
    candidates = []
    
    # 从验证集中获取一些样本
    for i, batch in enumerate(val_loader):
        if i >= num_examples:
            break
        src_ids = batch['input_ids']
        tgt_ids = batch['labels']
        
        src_text = tokenizer.decode(src_ids[0], skip_special_tokens=True)
        reference_text = tokenizer.decode(tgt_ids[0], skip_special_tokens=True)
        
        # ---> 核心修正: 使用 Beam Search 进行翻译 <---
        candidate_text = translate_sentence_beam_search(model, tokenizer, src_text, config, beam_size=3)
        
        references.append([reference_text.split()])
        candidates.append(candidate_text.split())

    if not candidates:
        return 0.0
        
    # 计算 BLEU-4 分数
    score = bleu_score(candidates, references, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    return score


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
        task_type=config["task_type"],
        # ---> 新增: 传递消融实验开关 <---
        use_pos_enc=config.get("use_positional_encoding", True),
        use_residual=config.get("use_residual_connections", True)
    ).to(device)

    # ---> 核心修正: 将 ignore_index 设置为 -100 <---
    # 这与 DataCollatorForSeq2Seq 的默认行为匹配，它用 -100 来填充标签。
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)

    if config.get("warmup_steps", 0) > 0:
        print(f"学习率预热将持续 {config['warmup_steps']} 步。")
    else:
        print("未设置学习率预热。")

    # 5. 初始化训练状态
    start_epoch = 0
    best_val_loss = float('inf')
    best_bleu_score = 0.0
    # ---> 新增: 全局步数，用于学习率预热 <---
    global_step = 0
    best_model_path = os.path.join(os.path.dirname(__file__), "best_model_checkpoint.pt")

    if args.resume and os.path.exists(best_model_path):
        print(f"🔄 从检查点恢复训练: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        
        # ---> 修正: 恢复所有相关指标，使用 .get 保证向后兼容 <---
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        best_bleu_score = checkpoint.get('bleu_score', 0.0)
        # ---> 新增: 恢复全局步数 <---
        global_step = checkpoint.get('global_step', 0)

        if 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            print("学习率调度器状态已加载。")

        print(f"已加载 epoch {start_epoch} 的检查点，将从 epoch {start_epoch + 1} 开始训练。")
        print(f"恢复的最佳指标: val_loss={best_val_loss:.4f}, bleu_score={best_bleu_score:.4f}, global_step={global_step}")

    # ---> 修正: 如果是翻译任务，也加载 tokenizer <---
    tokenizer = None
    if config['task_type'] == 'sequence_to_sequence':
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    print(f"--- 开始在 {device} 上训练 '{args.config}' ---")
    metrics_file = os.path.join(os.path.dirname(__file__), "results", f"metrics_{args.config}.csv")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    if not os.path.exists(metrics_file) or start_epoch == 0:
        with open(metrics_file, "w", encoding="utf-8") as f:
            # ---> 修正: 添加 bleu_score 列 <---
            f.write("epoch,train_loss,val_loss,train_ppl,val_ppl,learning_rate,bleu_score\n")

    for epoch in range(start_epoch, config["epochs"]):
        # ---> 修正: 传入并接收 global_step <---
        train_loss, global_step = train_one_epoch(model, train_loader, criterion, optimizer, config, epoch, global_step)
        
        val_loss = float('inf')
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, config)
        else:
            val_loss = train_loss 
            if epoch == 0:
                print("[警告] 未找到验证数据加载器，将基于训练损失保存模型。")

        # current_lr = scheduler.get_last_lr()[0] # <--- 删除旧的获取lr方式
        # scheduler.step() # <--- 删除旧的 scheduler step

        # --- 指标计算与记录 ---
        # 困惑度对所有任务都计算并保存，因为它直接来源于 loss
        train_ppl = math.exp(min(train_loss, 100.0))
        val_ppl = math.exp(min(val_loss, 100.0))
        
        # BLEU 分数只为翻译任务计算
        bleu = 0.0
        if config['task_type'] == 'sequence_to_sequence' and val_loader:
            bleu = calculate_bleu(model, tokenizer, val_loader, config)

        # ---> 修正: 从优化器直接获取当前学习率用于记录 <---
        current_lr = optimizer.param_groups[0]['lr']

        # --- 打印日志 (根据任务类型，突出不同重点) ---
        if config['task_type'] == 'sequence_to_sequence':
            # ---> 修正: 按要求在训练日志中加入困惑度 PPL <---
            # 对于翻译任务，我们更关注 BLEU 分数
            print(f"Epoch {epoch+1}/{config['epochs']}: train_loss={train_loss:.4f} (PPL: {train_ppl:.2f}), val_loss={val_loss:.4f} (PPL: {val_ppl:.2f}), BLEU={bleu:.4f}, lr={current_lr:.6f}")
        else: # 默认或 language_modeling
            # 对于其他任务，我们关注困惑度
            print(f"Epoch {epoch+1}/{config['epochs']}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}, lr={current_lr:.6f}")

        # --- 保存所有指标到 CSV ---
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{train_ppl:.6f},{val_ppl:.6f},{current_lr:.8f},{bleu:.6f}\n")

        # --- 保存最佳模型检查点 (根据任务类型选择标准) ---
        save_checkpoint = False
        if config['task_type'] == 'sequence_to_sequence':
            if bleu > best_bleu_score:
                best_bleu_score = bleu
                save_checkpoint = True
                print(f"🎉 新的最佳模型! BLEU 分数提升至: {bleu:.4f}。模型已保存至 -> {best_model_path}")
        else: # 对于其他任务，仍然使用 val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint = True
                print(f"🎉 新的最佳模型! 验证损失降低至: {val_loss:.4f}。模型已保存至 -> {best_model_path}")

        if save_checkpoint:
            # ---> 改进的保存逻辑 <---
            temp_model_path = best_model_path + ".tmp"
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                # 'scheduler_state': scheduler.state_dict(), # <--- 删除
                'val_loss': val_loss,
                'bleu_score': bleu,
                'config': config,
                'global_step': global_step # <--- 新增: 保存全局步数
            }, temp_model_path)
            
            # 只有在成功保存到临时文件后，才用它覆盖旧的检查点
            os.replace(temp_model_path, best_model_path)

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
