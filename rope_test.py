import torch
import math


def complex_rope_example():
    # 配置参数
    batch_size = 2
    num_heads = 4
    seq_length = 3
    head_dim = 8

    # 1. 创建更复杂的输入张量
    q = torch.randn(batch_size, num_heads, seq_length, head_dim)
    k = torch.randn(batch_size, num_heads, seq_length, head_dim)

    # 2. 计算实际的位置编码
    def get_rotary_embeddings(seq_length, dim, base=10000):
        # 计算不同维度的角度
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        # 计算每个位置的角度
        position = torch.arange(seq_length).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        # 计算sin和cos值
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        # 扩展维度以匹配注意力头
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
        # 复制到所有维度
        sin = sin.repeat(batch_size, num_heads, 1, 2)
        cos = cos.repeat(batch_size, num_heads, 1, 2)
        return cos, sin

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin):
        # 应用旋转位置编码
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # 3. 获取位置编码
    cos, sin = get_rotary_embeddings(seq_length, head_dim)

    # 4. 应用RoPE
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

    # 5. 计算注意力分数
    attention_scores = torch.matmul(q_embed, k_embed.transpose(
        -2, -1)) / math.sqrt(head_dim)

    # 打印详细信息
    print(f"输入形状:")
    print(f"q shape: {q.shape}")
    print(f"k shape: {k.shape}")
    print(f"\n位置编码形状:")
    print(f"cos shape: {cos.shape}")
    print(f"sin shape: {sin.shape}")

    # 展示第一个batch、第一个head的数据示例
    print(f"\n原始q (batch=0, head=0, pos=0):\n{q[0,0,0]}")
    print(f"\n旋转后q (batch=0, head=0, pos=0):\n{q_embed[0,0,0]}")

    # 展示不同位置的旋转效果
    print(f"\n位置0和位置1的q对比:")
    print(f"pos 0: {q_embed[0,0,0][:4]}")  # 只显示前4个值
    print(f"pos 1: {q_embed[0,0,1][:4]}")

    # 计算注意力分数
    print(f"\n注意力分数 (batch=0, head=0):\n{attention_scores[0,0]}")

    return q_embed, k_embed, attention_scores


# 运行示例并添加随机种子以确保可重复性
torch.manual_seed(42)
q_embed, k_embed, attention_scores = complex_rope_example()


# 分析注意力模式
def analyze_attention_pattern(attention_scores):
    """分析注意力分数的模式"""
    # 对每个头的注意力分数进行softmax
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

    print("\n注意力模式分析:")
    print("平均注意力分数:", attention_probs.mean().item())
    print("最大注意力分数:", attention_probs.max().item())
    print("最小注意力分数:", attention_probs.min().item())

    # 分析相对位置的注意力强度
    for i in range(attention_probs.size(1)):  # 对每个位置
        print(f"\n位置 {i} 的注意力分布:")
        print(attention_probs[0, 0, i])  # 第一个batch、第一个head


analyze_attention_pattern(attention_scores)
