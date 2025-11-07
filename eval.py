import torch
import torch.nn.functional as F # <--- 确保在文件顶部导入
from src.model import Transformer
# get_config 实际上不再是必需的，因为配置将从检查点加载，但保留它以防万一
from src.config import get_config 
from transformers import AutoTokenizer # <--- 确保导入 AutoTokenizer

def load_model_from_checkpoint(checkpoint_path):
    """
    从检查点加载模型和配置。
    
    这个函数现在使用保存在检查点文件中的配置来确保
    模型架构与训练时完全一致。
    """
    # 确定运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 首先加载检查点文件
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())

    # 2. 从检查点中提取训练时使用的配置
    # 这是最关键的修正：使用检查点中的配置来保证一致性
    if 'config' not in checkpoint:
        raise ValueError("错误：检查点文件中未找到 'config'。请使用较新的训练脚本重新训练或手动提供配置。")
    config = checkpoint['config']
    print("Config keys from checkpoint:", config.keys())

    # 3. 使用检查点中的配置来初始化模型
    # **重要：确保将 max_len 传递给模型！**
    model = Transformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"],
        task_type=config.get("task_type", "language_modeling") # <--- 新增: 传递任务类型
    ).to(device)
    
    # 4. 现在加载模型状态，尺寸会完全匹配
    model.load_state_dict(checkpoint['model_state'])

    # 检查 'stoi' 和 'itos' (字符到索引的映射) 是否存在，如果不存在则尝试生成
    if 'stoi' not in config or 'itos' not in config:
        print("警告：检查点中缺少 'stoi' 或 'itos'，尝试从本地文件手动生成...")
        try:
            # 确保这个路径是正确的
            with open("data/tinyshakespeare.txt", 'r', encoding='utf-8') as f:
                text = f.read()
            chars = sorted(list(set(text)))
            config['stoi'] = {ch: i for i, ch in enumerate(chars)}
            config['itos'] = {i: ch for i, ch in enumerate(chars)}
            print("已成功生成 'stoi' 和 'itos'。")
        except FileNotFoundError:
            print("错误：无法找到 data/tinyshakespeare.txt 文件来生成词汇表。")
            exit() # 如果没有映射关系，程序无法继续

    # 将设备信息也添加到配置中，方便 generate_text 函数使用
    config['device'] = device
    
    model.eval()  # 设置为评估模式
    # ---> 修正: 返回 tokenizer 以便在外部使用 <---
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    return model, config, tokenizer

def generate_text(model, config, start_text, max_length=100, temperature=0.8, top_k=20):
    """
    使用训练好的模型生成文本 (使用温度和 Top-K 采样)。
    """
    device = config["device"]
    stoi = config['stoi']
    itos = config['itos']

    # 将起始文本转换为 token IDs
    initial_tokens = [stoi[c] for c in start_text if c in stoi]
    if not initial_tokens:
        print(f"错误：起始文本 '{start_text}' 中的所有字符都不在词汇表中。")
        return ""
        
    input_ids = torch.tensor(initial_tokens, dtype=torch.long).unsqueeze(1).to(device)
    generated_tokens = initial_tokens.copy()

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # 确保输入序列不会超过模型的最大长度
            # 如果超过，我们只使用最后 max_len-1 个 token 作为上下文
            input_context = input_ids[-config['max_len']:]

            # 创建因果掩码
            causal_mask = torch.triu(torch.ones(input_context.size(0), input_context.size(0), device=device), diagonal=1).bool()

            # 获取模型输出
            output = model(src=input_context, tgt=input_context, src_mask=None, tgt_mask=causal_mask)

            # 1. 获取最后一个时间步的 logits
            next_token_logits = output[-1, 0, :]

            # 2. 应用温度，增加生成文本的多样性
            next_token_logits = next_token_logits / temperature

            # 3. (可选) 应用 Top-K 过滤，限制采样范围，防止生成不合逻辑的词
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')

            # 4. 从修改后的 logits 计算概率分布，并进行采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # 将新生成的 token 添加到序列中
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=0)

            # 如果模型定义了结束符，可以在这里添加逻辑
            if '<eos>' in itos and next_token == stoi.get('<eos>'):
                break

    return "".join([itos.get(token_id, '?') for token_id in generated_tokens])


# ---> 新增: 用于机器翻译的贪心解码函数 <---
def translate_sentence(model, tokenizer, sentence, config):
    """使用贪心解码翻译单个句子"""
    model.eval()
    device = config['device']
    
    # 1. 对源句子进行分词
    src_tokens = tokenizer.encode(sentence, return_tensors="pt").to(device)
    src_tokens = src_tokens.t() # [seq_len, 1]

    # ---> 新增: 为源句子创建 padding mask <---
    pad_token_id = tokenizer.pad_token_id
    # src_tokens shape: [seq_len, 1], mask shape: [1, seq_len]
    src_padding_mask = (src_tokens == pad_token_id).transpose(0, 1)

    # 2. 编码源句子，并传入 padding mask
    with torch.no_grad():
        # ---> 修正: 将 src_padding_mask 传递给 encoder <---
        memory = model.encode(src_tokens, src_mask=src_padding_mask)

    # 3. 解码过程
    # ---> 核心修正: 使用更通用的 BOS token 作为解码起始信号 <---
    # 检查 tokenizer 是否有 bos_token_id，如果没有，则回退到 pad_token_id
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
    tgt_tokens = [start_token]
    
    for _ in range(config['max_len']):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(1).to(device) # [current_len, 1]
        tgt_mask = torch.triu(torch.ones(len(tgt_tokens), len(tgt_tokens), device=device), diagonal=1).bool()

        with torch.no_grad():
            # ---> 修正: 将 src_padding_mask 作为 memory_mask 传递给 decoder <---
            output = model.decode(tgt_tensor, memory, tgt_mask, memory_mask=src_padding_mask)
            # output: [current_len, 1, vocab_size]
            next_token_logits = output[-1, 0, :]
            next_token = next_token_logits.argmax().item()
        
        tgt_tokens.append(next_token)

        # 如果生成了 EOS token，则停止
        if next_token == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(tgt_tokens, skip_special_tokens=True)


# ---> 新增: 功能更强大的 Beam Search 解码函数 <---
def translate_sentence_beam_search(model, tokenizer, sentence, config, beam_size=3, max_len=None):
    """使用束搜索 (Beam Search) 翻译单个句子"""
    model.eval()
    device = config['device']
    max_len = max_len if max_len is not None else config['max_len']

    # 1. 编码源句子
    src_tokens = tokenizer.encode(sentence, return_tensors="pt").to(device).t()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    src_padding_mask = (src_tokens == pad_token_id).transpose(0, 1)
    with torch.no_grad():
        memory = model.encode(src_tokens, src_mask=src_padding_mask)

    # 2. 初始化束
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else pad_token_id
    beams = [(torch.tensor([start_token], device=device), 0.0)]
    completed_beams = []

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1].item() == tokenizer.eos_token_id:
                completed_beams.append((seq, score))
                continue

            tgt_tensor = seq.unsqueeze(1)
            tgt_mask = torch.triu(torch.ones(len(seq), len(seq), device=device), diagonal=1).bool()
            
            with torch.no_grad():
                output = model.decode(tgt_tensor, memory, tgt_mask, memory_mask=src_padding_mask)
                logits = output[-1, 0, :]
                log_probs = F.log_softmax(logits, dim=-1)

            top_log_probs, top_indices = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_token = top_indices[i].unsqueeze(0)
                new_seq = torch.cat([seq, next_token])
                new_score = score + top_log_probs[i].item()
                new_beams.append((new_seq, new_score))

        if not new_beams:
            break
        
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if all(b[0][-1].item() == tokenizer.eos_token_id for b in beams):
            completed_beams.extend(beams)
            break

    completed_beams.extend(beams)
    if not completed_beams: return ""

    best_beam = sorted(completed_beams, key=lambda x: x[1] / len(x[0]), reverse=True)[0]
    best_seq = best_beam[0].tolist()
    
    return tokenizer.decode(best_seq, skip_special_tokens=True)


# 主执行函数
if __name__ == "__main__":
    checkpoint_path = "best_model_checkpoint.pt"
    
    print(f"--- 正在从 '{checkpoint_path}' 加载模型 ---")
    # ---> 修正: 接收 tokenizer <---
    model, config, tokenizer = load_model_from_checkpoint(checkpoint_path)
    print("--- 模型加载成功 ---")

    # ---> 新增代码: 计算并打印模型参数量 <---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params/1e6:.2f} M")
    print("--------------------")

    # 定义生成文本的起始句
    start_text = "To be, or not to be, that is the question:"

    # 生成文本
    generated_text = generate_text(
        model=model, 
        config=config, 
        start_text=start_text, 
        max_length=300,      # 生成的最大长度
        temperature=1.2,     # 温度参数，值越高，随机性越大
        top_k=50             # Top-K 采样，限制备选词的数量
    )
    
    print("\n--- Generated Text ---")
    print(generated_text)


    #     print(f"Translated: {translation}")
