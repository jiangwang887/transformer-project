# import os
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from datasets import load_dataset
# from transformers import AutoTokenizer
# import requests

# def get_dataloaders(config):
#     """根据配置加载、处理并返回训练和验证 DataLoader"""
    
#     device = torch.device(config["device"])
    
#     # 如果用户指定了本地数据路径，优先使用本地文件（支持 tiny_shakespeare 的纯文本）
#     local_path = config.get('local_dataset_path', None)
#     if local_path:
#         local_path = os.path.expanduser(local_path)
#         if os.path.isfile(local_path):
#             print(f"检测到本地数据文件: {local_path}，将从本地加载（降级处理为 tiny_shakespeare 文本格式）...")
#             with open(local_path, 'r', encoding='utf-8') as f:
#                 full_text = f.read()
#             cut = int(len(full_text) * 0.9)
#             train_text = full_text[:cut]
#             val_text = full_text[cut:]
#             dataset = {
#                 'train': [{'text': train_text}],
#                 'validation': [{'text': val_text}]
#             }
#         else:
#             raise FileNotFoundError(f"指定的本地数据文件不存在: {local_path}")

#     else:
#         # --- 步骤 1: 加载数据集 ---
#         print(f"加载数据集 '{config['dataset_name']}'...")
#         try:
#             # 不再使用 trust_remote_code（已废弃/不支持）
#             dataset = load_dataset(config['dataset_name'], config.get('dataset_config'))
#         except RuntimeError as e:
#             # 兼容 tiny_shakespeare 的降级处理（datasets 不再支持脚本式数据集）
#             msg = str(e)
#             if 'tiny_shakespeare' in config['dataset_name'] or 'karpathy/tiny_shakespeare' in config['dataset_name'] or 'tiny_shakespeare.py' in msg:
#                 print("检测到 tiny_shakespeare 数据集脚本不被支持，改为从 karpathy GitHub raw 下载文本作为降级处理...")
#                 url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#                 r = requests.get(url, timeout=20)
#                 r.raise_for_status()
#                 full_text = r.text
#                 # 90% 作为 train，10% 作为 validation（按字符切分，并放入单条记录以兼容后续处理）
#                 cut = int(len(full_text) * 0.9)
#                 train_text = full_text[:cut]
#                 val_text = full_text[cut:]
#                 dataset = {
#                     'train': [{'text': train_text}],
#                     'validation': [{'text': val_text}]
#                 }
#             else:
#                 # 不是 tiny_shakespeare 的脚本问题，向上抛出以便用户处理
#                 raise

#     # --- 步骤 2: 初始化分词器和处理数据 ---
#     if config['task_type'] == 'language_modeling':
#         # 字符级分词器
#         text = "".join([example[config['text_col']] for example in dataset['train']])
#         chars = sorted(list(set(text)))
#         vocab_size = len(chars)
#         stoi = {ch: i for i, ch in enumerate(chars)}
#         itos = {i: ch for i, ch in enumerate(chars)}
#         encode = lambda s: [stoi[c] for c in s]
        
#         train_data = torch.tensor(encode("".join(d[config['text_col']] for d in dataset['train'])), dtype=torch.long)
#         val_data = torch.tensor(encode("".join(d[config['text_col']] for d in dataset['validation'])), dtype=torch.long)

#         # 创建 (X, Y) 数据对
#         def create_lm_dataset(data_tensor, block_size):
#             # 保证有足够的 token 用于输入和对应的右移目标
#             num_sequences = (len(data_tensor) - 1) // block_size
#             if num_sequences <= 0:
#                 raise ValueError("数据太短，请增大数据或减小 max_len")
#             data_tensor = data_tensor[: num_sequences * block_size + 1]  # +1 用于 targets 的右移
#             X = data_tensor[:-1].view(num_sequences, block_size)
#             Y = data_tensor[1:].view(num_sequences, block_size)
#             return TensorDataset(X, Y)

#         train_dataset = create_lm_dataset(train_data, config['max_len'])
#         val_dataset = create_lm_dataset(val_data, config['max_len'])

#         # 更新词汇表大小
#         config['vocab_size'] = vocab_size

#     elif config['task_type'] == 'sequence_to_sequence':
#         tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        
#         def tokenize_fn(examples):
#             inputs = tokenizer(examples[config['src_lang_col']], max_length=config['max_len'], truncation=True, padding="max_length")
#             labels = tokenizer(text_target=examples[config['tgt_lang_col']], max_length=config['max_len'], truncation=True, padding="max_length")
#             inputs['labels'] = labels['input_ids']
#             return inputs

#         tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)
#         tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

#         train_dataset = tokenized_datasets['train']
#         val_dataset = tokenized_datasets['validation']
        
#         config['vocab_size'] = tokenizer.vocab_size
#         config['pad_token_id'] = tokenizer.pad_token_id

#     elif config['task_type'] == 'text_classification':
#         tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        
#         def tokenize_fn(examples):
#             return tokenizer(examples[config['text_col']], max_length=config['max_len'], truncation=True, padding="max_length")

#         tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=[config['text_col']])
#         tokenized_datasets = tokenized_datasets.rename_column(config['label_col'], 'labels')
#         tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
#         train_dataset = tokenized_datasets['train']
#         # AG News 没有验证集，我们从测试集里分一个
#         split_dataset = tokenized_datasets['test'].train_test_split(test_size=0.5)
#         val_dataset = split_dataset['train']
#         # test_dataset = split_dataset['test']

#         config['vocab_size'] = tokenizer.vocab_size
#         config['pad_token_id'] = tokenizer.pad_token_id

#     else:
#         raise ValueError(f"不支持的任务类型: {config['task_type']}")

#     # --- 步骤 3: 创建 DataLoader ---
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
#     config['vocab_size'] = len(stoi)  # 在训练时动态生成词汇表大小
#     config['stoi'] = stoi
#     config['itos'] = itos
#     return train_loader, val_loader, config





import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import requests

def get_dataloaders(config):
    """根据配置加载、处理并返回训练和验证 DataLoader"""
    
    device = torch.device(config["device"])
    
    # 如果用户指定了本地数据路径，优先使用本地文件（支持 tiny_shakespeare 的纯文本）
    local_path = config.get('local_dataset_path', None)
    if local_path:
        local_path = os.path.expanduser(local_path)
        if os.path.isfile(local_path):
            print(f"检测到本地数据文件: {local_path}，将从本地加载（降级处理为 tiny_shakespeare 文本格式）...")
            with open(local_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            cut = int(len(full_text) * 0.9)
            train_text = full_text[:cut]
            val_text = full_text[cut:]
            dataset = {
                'train': [{'text': train_text}],
                'validation': [{'text': val_text}]
            }
        else:
            raise FileNotFoundError(f"指定的本地数据文件不存在: {local_path}")

    else:
        # --- 步骤 1: 加载数据集 ---
        print(f"加载数据集 '{config['dataset_name']}'...")
        try:
            # 不再使用 trust_remote_code（已废弃/不支持）
            dataset = load_dataset(config['dataset_name'], config.get('dataset_config'))
        except RuntimeError as e:
            # 兼容 tiny_shakespeare 的降级处理（datasets 不再支持脚本式数据集）
            msg = str(e)
            if 'tiny_shakespeare' in config['dataset_name'] or 'karpathy/tiny_shakespeare' in config['dataset_name'] or 'tiny_shakespeare.py' in msg:
                print("检测到 tiny_shakespeare 数据集脚本不被支持，改为从 karpathy GitHub raw 下载文本作为降级处理...")
                url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                full_text = r.text
                # 90% 作为 train，10% 作为 validation（按字符切分，并放入单条记录以兼容后续处理）
                cut = int(len(full_text) * 0.9)
                train_text = full_text[:cut]
                val_text = full_text[cut:]
                dataset = {
                    'train': [{'text': train_text}],
                    'validation': [{'text': val_text}]
                }
            else:
                # 不是 tiny_shakespeare 的脚本问题，向上抛出以便用户处理
                raise

    # --- 步骤 2: 初始化分词器和处理数据 ---
    if config['task_type'] == 'language_modeling':
        # 字符级分词器
        text = "".join([example[config['text_col']] for example in dataset['train']])
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        
        train_data = torch.tensor(encode("".join(d[config['text_col']] for d in dataset['train'])), dtype=torch.long)
        val_data = torch.tensor(encode("".join(d[config['text_col']] for d in dataset['validation'])), dtype=torch.long)

        # 创建 (X, Y) 数据对
        def create_lm_dataset(data_tensor, block_size):
            # 保证有足够的 token 用于输入和对应的右移目标
            num_sequences = (len(data_tensor) - 1) // block_size
            if num_sequences <= 0:
                raise ValueError("数据太短，请增大数据或减小 max_len")
            data_tensor = data_tensor[: num_sequences * block_size + 1]  # +1 用于 targets 的右移
            X = data_tensor[:-1].view(num_sequences, block_size)
            Y = data_tensor[1:].view(num_sequences, block_size)
            return TensorDataset(X, Y)

        train_dataset = create_lm_dataset(train_data, config['max_len'])
        val_dataset = create_lm_dataset(val_data, config['max_len'])

        # 更新词汇表大小
        config['vocab_size'] = vocab_size
        # --- 修正 3: 将 stoi/itos 的赋值移到这里，确保逻辑闭环 ---
        config['stoi'] = stoi
        config['itos'] = itos

    elif config['task_type'] == 'sequence_to_sequence':
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        
        def tokenize_fn(examples):
            inputs = tokenizer(examples[config['src_lang_col']], max_length=config['max_len'], truncation=True, padding="max_length")
            labels = tokenizer(text_target=examples[config['tgt_lang_col']], max_length=config['max_len'], truncation=True, padding="max_length")
            inputs['labels'] = labels['input_ids']
            return inputs

        tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        train_dataset = tokenized_datasets['train']
        val_dataset = tokenized_datasets['validation']
        
        config['vocab_size'] = tokenizer.vocab_size
        config['pad_token_id'] = tokenizer.pad_token_id

    elif config['task_type'] == 'text_classification':
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        
        def tokenize_fn(examples):
            return tokenizer(examples[config['text_col']], max_length=config['max_len'], truncation=True, padding="max_length")

        tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=[config['text_col']])
        tokenized_datasets = tokenized_datasets.rename_column(config['label_col'], 'labels')
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        train_dataset = tokenized_datasets['train']
        # AG News 没有验证集，我们从测试集里分一个
        split_dataset = tokenized_datasets['test'].train_test_split(test_size=0.5)
        val_dataset = split_dataset['train']
        # test_dataset = split_dataset['test']

        config['vocab_size'] = tokenizer.vocab_size
        config['pad_token_id'] = tokenizer.pad_token_id

    else:
        raise ValueError(f"不支持的任务类型: {config['task_type']}")

    # --- 步骤 3: 创建 DataLoader ---
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    # --- 修正 3: 从此处删除错误的全局赋值 ---
    return train_loader, val_loader, config