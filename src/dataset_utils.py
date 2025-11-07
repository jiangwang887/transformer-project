# filepath: f:\transformer-project\src\dataset_utils.py
import os
import torch
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
# ---> 核心修正: 导入为 Seq2Seq 任务专门设计的 DataCollator <---
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

def get_dataloaders(config):
    """
    根据配置加载数据，支持从 Hugging Face Hub 或本地并行文件目录加载。
    """
    print("--- 正在准备数据加载器 ---")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config['pad_token_id'] = tokenizer.pad_token_id
    config['vocab_size'] = tokenizer.vocab_size

    raw_datasets = None

    if config.get("local_dataset_path"):
        local_path = config["local_dataset_path"]
        print(f"检测到本地数据集配置，路径: {local_path}")

        if os.path.isdir(local_path):
            print("路径是一个目录，正在尝试加载并行文件 (train/val)...")
            try:
                train_src_file = os.path.join(local_path, f"train.{config['src_lang']}")
                train_tgt_file = os.path.join(local_path, f"train.{config['tgt_lang']}")
                # ---> 修正: 使用 'val' 而不是 'valid' 来匹配您的文件名 <---
                val_src_file = os.path.join(local_path, f"valid.{config['src_lang']}")
                val_tgt_file = os.path.join(local_path, f"valid.{config['tgt_lang']}")

                with open(train_src_file, 'r', encoding='utf-8') as f:
                    train_src_texts = [line.strip() for line in f]
                with open(train_tgt_file, 'r', encoding='utf-8') as f:
                    train_tgt_texts = [line.strip() for line in f]
                
                with open(val_src_file, 'r', encoding='utf-8') as f:
                    val_src_texts = [line.strip() for line in f]
                with open(val_tgt_file, 'r', encoding='utf-8') as f:
                    val_tgt_texts = [line.strip() for line in f]

                train_data = {'translation': [{config['src_lang']: en, config['tgt_lang']: de} for en, de in zip(train_src_texts, train_tgt_texts)]}
                val_data = {'translation': [{config['src_lang']: en, config['tgt_lang']: de} for en, de in zip(val_src_texts, val_tgt_texts)]}
                
                train_dataset = Dataset.from_dict(train_data)
                val_dataset = Dataset.from_dict(val_data)
                
                raw_datasets = DatasetDict({
                    'train': train_dataset,
                    'validation': val_dataset
                })
                print(f"成功从目录 {local_path} 加载 {len(raw_datasets['train'])} 条训练数据和 {len(raw_datasets['validation'])} 条验证数据。")

            except FileNotFoundError as e:
                print(f"错误: 在目录 {local_path} 中找不到预期的文件。")
                raise e
        else:
            raise FileNotFoundError(f"指定的本地数据路径不是一个有效的目录: {local_path}")
    
    if raw_datasets is None:
        print(f"未找到或加载本地数据失败，将从 Hugging Face Hub 下载 '{config['dataset_name']}'...")
        raw_datasets = load_dataset(config['dataset_name'], config.get('dataset_config_name'))

    def tokenize_fn(examples):
        inputs = [ex[config['src_lang']] for ex in examples['translation']]
        targets = [ex[config['tgt_lang']] for ex in examples['translation']]
        
        model_inputs = tokenizer(inputs, max_length=config['max_len'], truncation=True)
        labels = tokenizer(targets, max_length=config['max_len'], truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True, remove_columns=raw_datasets["train"].column_names)
    
    # ---> 核心修正: 使用 DataCollatorForSeq2Seq <---
    # 这个整理器专门为序列到序列任务设计，能正确处理输入和标签的填充。
    # 它会自动用 -100 填充标签，这个值会被损失函数标准地忽略。
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # 创建 DataLoader
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_loader = None
    if "validation" in tokenized_datasets:
        val_loader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=data_collator
        )

    return train_loader, val_loader, config
