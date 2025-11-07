import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    为输入序列的 token 注入位置信息。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    实现多头注意力机制。
    输入/输出均为 shape [seq_len, batch_size, d_model]
    mask 语义：True 表示被屏蔽的位置（会被置为 -inf）
    mask 可以是 [seq_len, seq_len] 或 [batch_size, seq_len, seq_len]
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])  # q,k,v,out
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # query/key/value: [seq_len, batch, d_model]
        seq_len_q, batch_size, _ = query.size()
        seq_len_k = key.size(0)

        # 线性变换并 reshape -> [batch, heads, seq_len, d_k]
        q = self.linears[0](query).view(seq_len_q, batch_size, self.num_heads, self.d_k).permute(1,2,0,3)
        k = self.linears[1](key).view(seq_len_k, batch_size, self.num_heads, self.d_k).permute(1,2,0,3)
        v = self.linears[2](value).view(seq_len_k, batch_size, self.num_heads, self.d_k).permute(1,2,0,3)

        # scores: [batch, heads, seq_q, seq_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ---> 修正: 改进 mask 处理逻辑以支持 padding mask <---
        if mask is not None:
            # mask: True 表示要屏蔽的位置
            # 预处理 mask 以便广播到 scores [batch, heads, seq_q, seq_k]
            if mask.dim() == 2:
                # 两种情况:
                # 1. padding mask, shape [batch, seq_k] -> unsqueeze to [batch, 1, 1, seq_k]
                # 2. causal mask, shape [seq_q, seq_k] -> unsqueeze to [1, 1, seq_q, seq_k]
                if mask.size(0) == batch_size: # 启发式判断: 如果第一维是 batch_size, 则是 padding mask
                    mask = mask.unsqueeze(1).unsqueeze(2)
                else: # 否则认为是 causal mask
                    mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                # 假设是 [batch, seq_q, seq_k], 增加 head 维度
                # [batch, seq_q, seq_k] -> [batch, 1, seq_q, seq_k]
                mask = mask.unsqueeze(1)
            # mask 此时应为 4D, 可以广播
            scores = scores.masked_fill(mask, float('-inf'))

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, v)  # [batch, heads, seq_q, d_k]
        # concat heads -> [seq_q, batch, d_model]
        x = x.permute(2,0,1,3).contiguous().view(seq_len_q, batch_size, self.num_heads * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """
    实现 FFN (Feed-Forward Network)。
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model] -> linear handles last dim
        return self.w2(self.dropout(self.activation(self.w1(x))))

# model.py

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, use_residual: bool):
        super().__init__()
        self.use_residual = use_residual
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # ---> 修正: 使用 Pre-Layer Normalization <---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ---> 修正: Pre-LN 结构 <---
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, mask)
        if self.use_residual:
            x = x + self.dropout(attn_output)
        else:
            x = self.dropout(attn_output)

        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        if self.use_residual:
            x = x + self.dropout(ff_output)
        else:
            x = self.dropout(ff_output)
            
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, use_residual: bool):
        super().__init__()
        self.use_residual = use_residual
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # ---> 修正: 使用 Pre-Layer Normalization <---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        # ---> 修正: Pre-LN 结构 <---
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)
        if self.use_residual:
            x = x + self.dropout(attn_output)
        else:
            x = self.dropout(attn_output)

        if memory is not None:
            norm_x = self.norm2(x)
            cross_attn_output = self.cross_attn(norm_x, memory, memory, memory_mask)
            if self.use_residual:
                x = x + self.dropout(cross_attn_output)
            else:
                x = self.dropout(cross_attn_output)

        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        if self.use_residual:
            x = x + self.dropout(ff_output)
        else:
            x = self.dropout(ff_output)
            
        return x

class Transformer(nn.Module):
    """
    一个标准的 Transformer 模型（包含 embedding + pos enc）。
    接受 src/tgt 为 [seq_len, batch] 的 token ids，返回 logits: [seq_len, batch, vocab_size]
    """
 
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1, max_len: int = 512, task_type: str = 'sequence_to_sequence', use_pos_enc: bool = True, use_residual: bool = True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_len)
        # ---> 修正: 将开关传递给 Block <---
        self.layers_enc = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout, use_residual) for _ in range(num_layers)])
        self.layers_dec = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout, use_residual) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.task_type = task_type
        self.use_pos_enc = use_pos_enc # <--- 保存开关状态

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src_emb = self.token_emb(src) * math.sqrt(self.d_model)
        if self.use_pos_enc:
            src_emb = self.pos_enc(src_emb)
        memory = src_emb
        for layer in self.layers_enc:
            memory = layer(memory, src_mask)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        tgt_emb = self.token_emb(tgt) * math.sqrt(self.d_model)
        if self.use_pos_enc:
            tgt_emb = self.pos_enc(tgt_emb)
        output = tgt_emb
        for layer in self.layers_dec:
        
            output = layer(output, memory, tgt_mask, memory_mask)
        logits = self.out_proj(output)
        return logits

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        tgt_emb = self.token_emb(tgt) * math.sqrt(self.d_model)
    
        if self.use_pos_enc:
            tgt_emb = self.pos_enc(tgt_emb)

        memory = None
        if self.task_type != 'language_modeling':
            src_emb = self.token_emb(src) * math.sqrt(self.d_model)
      
            if self.use_pos_enc:
                src_emb = self.pos_enc(src_emb)
            memory = src_emb
            for layer in self.layers_enc:
                memory = layer(memory, src_mask)

        output = tgt_emb
        for layer in self.layers_dec:
            # note: pass src_mask as memory_mask (encoder-decoder mask)
            # 如果 memory 为 None, DecoderBlock 内部会跳过交叉注意力
       
            output = layer(output, memory, tgt_mask, src_mask)

        # output: [seq_len, batch, d_model] -> logits [seq_len, batch, vocab]
        logits = self.out_proj(output)
        return logits
