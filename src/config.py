import torch
def get_config(name="tiny_shakespeare"):
    """
    返回一个包含模型和训练超参数的配置字典。
    """
    base_config = {
        "device": "cuda",
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 1e-4,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_len": 512,
        "use_positional_encoding": True,
        "use_residual_connections": True,
        "warmup_steps": 4000,
        "local_dataset_path": None,
    }

    if name == "tiny_shakespeare":
        task_config = {
            "task_type": "language_modeling",
            "dataset_name": "tiny_shakespeare",
            "tokenizer_name": "gpt2",
            "learning_rate": 1e-3,
            "epochs": 5,
            "d_model": 256,
            "num_heads": 4,
            "num_layers": 4,
            "d_ff": 1024,
        }
        base_config.update(task_config)
        return base_config

    if name == "iwslt2017":
        task_config = {
            "task_type": "sequence_to_sequence",
            "dataset_name": "iwslt2017",
            "dataset_config_name": "iwslt2017-en-de",
            "src_lang": "en",
            "tgt_lang": "de",
            # ---> 最终、唯一的修正: 指向您本地的分词器目录 <---
            # 根据您的截图，这个目录就在您的项目根目录下
            "tokenizer_name": "./Helsinki-NLP-opus-mt-en-de",
            "learning_rate": 3e-4,
            "epochs": 15,
            "local_dataset_path": "./data/iwslt2017_local_text", 
        }
        base_config.update(task_config)
        return base_config

    raise ValueError(f"未知的配置名称: {name}")
