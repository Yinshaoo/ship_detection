import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class NSA(BaseModule):
    def __init__(
        self,
        embed_dims=96,
        num_heads=8,
        topk=4,
        block_size=7,
        swin_size=7,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        proj_drop_rate=0.,
        act_cfg=dict(type='GELU'),
        batch_first=True
    ):
        super().__init__()
        # 嵌入维度
        self.embed_dims = embed_dims
        # 注意力头的数量
        self.num_heads = num_heads
        # 块大小
        self.block_size = block_size
        # 滑动窗口大小
        self.swin_size = swin_size
        # 选择的 top-k 块数量
        self.topk = topk

        # 共享组件
        # 用于生成查询（Q）、键（K）和值（V）的线性层
        self.qkv = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        # 用于压缩块的线性层
        self.cmp = nn.Linear(block_size, 1)
        # 缩放因子
        self.scale = qk_scale or (embed_dims // num_heads) ** -0.5
        # 注意力 dropout 层
        self.attn_drop = nn.Dropout(attn_drop_rate)
        # 投影层
        self.proj = nn.Linear(embed_dims, embed_dims)
        # 投影后的 dropout 层
        self.proj_drop = nn.Dropout(proj_drop_rate)
        # 注意力 softmax 层
        self.softmax = nn.Softmax(dim=-1)
        # 门控线性层
        self.gate = nn.Linear(embed_dims, 3)

    def _prepare_qkv(self, x):
        """
        准备查询（Q）、键（K）和值（V）。
        :param x: 输入张量，形状为 [B, L, C]
        :return: Q、K、V 张量，形状均为 [B, nH, L, C//nH]
        """
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = k = v = qkv
        return q, k, v

    def _pad_and_reshape(self, k, v):
        """
        对键（K）和值（V）进行填充和重塑。
        :param k: 键张量，形状为 [B, nH, L, C//nH]
        :param v: 值张量，形状为 [B, nH, L, C//nH]
        :return: 填充和重塑后的 K 和 V 张量
        """
        B, nH, L, C_ = k.shape
        pad_L = (self.block_size - L % self.block_size) % self.block_size
        k_pad = F.pad(k, (0, 0, 0, pad_L))
        v_pad = F.pad(v, (0, 0, 0, pad_L))
        k_pad = k_pad.reshape(B, nH, -1, self.block_size, C_)
        v_pad = v_pad.reshape(B, nH, -1, self.block_size, C_)
        return k_pad, v_pad

    def _compute_attn(self, q, k, scale):
        """
        计算注意力分数。
        :param q: 查询张量，形状为 [B, nH, L, C//nH]
        :param k: 键张量，形状为 [B, nH, ..., C//nH]
        :param scale: 缩放因子
        :return: 注意力分数张量
        """
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        return attn

    def token_compression(self, q, k, v):
        """
        令牌压缩注意力机制。
        :param q: 查询张量，形状为 [B, nH, L, C//nH]
        :param k: 键张量，形状为 [B, nH, L, C//nH]
        :param v: 值张量，形状为 [B, nH, L, C//nH]
        :return: 压缩后的输出张量，形状为 [B, L, C]
        """
        B, nH, L, C_ = q.shape
        k_pad, v_pad = self._pad_and_reshape(k, v)
        k_cmp = self.cmp(k_pad.permute(0, 1, 2, 4, 3)).squeeze(-1)
        v_cmp = self.cmp(v_pad.permute(0, 1, 2, 4, 3)).squeeze(-1)
        attn = self._compute_attn(q, k_cmp, self.scale)
        x = (attn @ v_cmp).transpose(1, 2).reshape(B, L, C_ * nH)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def token_selection(self, q, k, v):
        """
        令牌选择注意力机制。
        :param q: 查询张量，形状为 [B, nH, L, C//nH]
        :param k: 键张量，形状为 [B, nH, L, C//nH]
        :param v: 值张量，形状为 [B, nH, L, C//nH]
        :return: 选择后的输出张量，形状为 [B, L, C]
        """
        B, nH, L, C_ = q.shape
        scores = k.mean(dim=-1)
        _, idx = torch.topk(scores, self.topk, dim=-1)
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, C_)
        k_slc = k.gather(2, idx)
        v_slc = v.gather(2, idx)
        attn = self._compute_attn(q, k_slc, self.scale)
        x = (attn @ v_slc).transpose(1, 2).reshape(B, L, C_ * nH)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def sliding_window(self, q, k, v):
        """
        滑动窗口注意力机制。
        :param q: 查询张量，形状为 [B, nH, L, C//nH]
        :param k: 键张量，形状为 [B, nH, L, C//nH]
        :param v: 值张量，形状为 [B, nH, L, C//nH]
        :return: 滑动窗口后的输出张量，形状为 [B, L, C]
        """
        B, nH, L, C_ = q.shape
        k_swin = k[:, :, -self.swin_size:, :]
        v_swin = v[:, :, -self.swin_size:, :]
        attn = self._compute_attn(q, k_swin, self.scale)
        x = (attn @ v_swin).transpose(1, 2).reshape(B, L, C_ * nH)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, query_pos=None):
        """
        前向传播方法。
        :param x: 输入张量，形状为 [B, L, C]
        :param query_pos: 查询位置编码，可选
        :return: 最终输出张量，形状为 [B, L, C]
        """
        if query_pos is not None:
            x = x + query_pos
        q, k, v = self._prepare_qkv(x)
        x1 = self.token_compression(q, k, v)
        x2 = self.token_selection(q, k, v)
        x3 = self.sliding_window(q, k, v)
        out = torch.stack([x1, x2, x3], dim=-1)
        gate_weights = self.gate(x)
        # 对门控权重进行 softmax 操作
        gate_weights = self.softmax(gate_weights).unsqueeze(-1)
        out = torch.matmul(out, gate_weights).squeeze(-1)
        return out    