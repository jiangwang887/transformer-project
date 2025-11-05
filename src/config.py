
def get_config(name):
    """根据名称返回数据集和模型的配置"""
    
    # ------------------ 配置 1: Tiny Shakespeare (语言建模) ------------------
    if name == "tiny_shakespeare":
        return {
            "task_type": "language_modeling",
            "dataset_name": "karpathy/tiny_shakespeare",
            "dataset_config": None,
            "text_col": "text",
            "tokenizer_type": "char", # 'char' 或 'pretrained'
             "vocab_size": 65,  # 替换为你的词汇表大小
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 6,
            "d_ff": 1024,
            "dropout": 0.1,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "epochs": 120,
            "max_len": 256, # 也叫 block_size 或 context_window
            "device": "cuda"
        }
        
    # ------------------ 配置 2: IWSLT2017 (英德翻译) ------------------
    if name == "iwslt2017":
        return {
            "task_type": "sequence_to_sequence",
            "dataset_name": "iwslt2017",
            "dataset_config": "iwslt2017-de-en", # 指定语言对
            "src_lang_col": "de", # 源语言列名（在数据集中）
            "tgt_lang_col": "en", # 目标语言列名
            "tokenizer_type": "pretrained",
            "tokenizer_name": "t5-small", # 一个轻量级的预训练分词器

            "d_model": 256,
            "num_heads": 8,
            "num_layers": 6,
            "d_ff": 1024,
            "dropout": 0.1,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "epochs": 15,
            "max_len": 128,
            "device": "cuda"
        }

    # ------------------ 配置 3: AG News (文本分类) ------------------
    if name == "ag_news":
        return {
            "task_type": "text_classification",
            "dataset_name": "ag_news",
            "dataset_config": None,
            "text_col": "text",
            "label_col": "label",
            "num_classes": 4, # AG News 有 4 个类别
            "tokenizer_type": "pretrained",
            "tokenizer_name": "bert-base-uncased",

            "d_model": 256,
            "num_heads": 4,
            "num_layers": 4,
            "d_ff": 512,
            "dropout": 0.1,
            "batch_size": 64,
            "learning_rate": 2e-5, # 微调时学习率通常更小
            "epochs": 5,
            "max_len": 128,
            "device": "cuda"
        }
        
    raise ValueError(f"未知的配置名称: {name}")