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
        self.cmp = nn.MaxPool2d(kernel_size=self.block_size, stride=self.block_size, padding=0)
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
        B, nH, h, w, C_ = k.shape
        # 计算需要填充的数量
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size

        # 进行填充
        k_pad = F.pad(k, (0, 0, 0, pad_w, 0, pad_h))
        v_pad = F.pad(v, (0, 0, 0, pad_w, 0, pad_h))

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

    def token_compression(self, q, k, v,h,w):
        """
        令牌压缩注意力机制。
        :param q: 查询张量，形状为 [B, nH, L, C//nH]
        :param k: 键张量，形状为 [B, nH, L, C//nH]
        :param v: 值张量，形状为 [B, nH, L, C//nH]
        :return: 压缩后的输出张量，形状为 [B, L, C]
        """
        B, nH, L, C_ = q.shape
        k=k.permute(0, 1, 3, 2).reshape(B, nH*C_, h, w)
        v=v.permute(0, 1, 3, 2).reshape(B, nH*C_, h, w)
        k_cmp=self.cmp(k).reshape(B, nH, C_, -1).permute(0, 1, 3, 2)
        v_cmp=self.cmp(v).reshape(B, nH, C_, -1).permute(0, 1, 3, 2)
    
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

    def sliding_window(self, q, k, v, h, w):
        """
        滑动窗口注意力机制。
        :param q: 查询张量，形状为 [B, nH, L, C//nH]
        :param k: 键张量，形状为 [B, nH, L, C//nH]
        :param v: 值张量，形状为 [B, nH, L, C//nH]
        :return: 滑动窗口后的输出张量，形状为 [B, L, C]
        """
        B, nH, L, C_ = q.shape
        k = k.reshape(B, nH, h, w, C_)
        v = v.reshape(B, nH, h, w, C_)

        out = []
        for i in range(h):
            for j in range(w):
                start_i = max(0, i - self.swin_size // 2)
                end_i = min(h, i + self.swin_size // 2 + 1)
                start_j = max(0, j - self.swin_size // 2)
                end_j = min(w, j + self.swin_size // 2 + 1)

                k_window = k[:, :, start_i:end_i, start_j:end_j].reshape(B, nH, -1, C_)
                v_window = v[:, :, start_i:end_i, start_j:end_j].reshape(B, nH, -1, C_)

                q_token = q[:, :, i * w + j].unsqueeze(2)
                attn = self._compute_attn(q_token, k_window, self.scale)
                x_token = (attn @ v_window).transpose(1, 2).reshape(B, 1, C_ * nH)
                out.append(x_token)

        out = torch.cat(out, dim=1)
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

    def forward(self, x, query_pos=None, h=None, w=None):
        """
        前向传播方法。
        :param x: 输入张量，形状为 [B, L, C]
        :param query_pos: 查询位置编码，可选
        :return: 最终输出张量，形状为 [B, L, C]
        """
        if query_pos is not None:
            x = x + query_pos
        q, k, v = self._prepare_qkv(x)
        x1 = self.token_compression(q, k, v, h, w)
        x2 = self.token_selection(q, k, v)
        x3 = self.sliding_window(q, k, v, h, w)
        out = torch.stack([x1, x2, x3], dim=-1)
        gate_weights = self.gate(x)
        # 对门控权重进行 softmax 操作
        gate_weights = self.softmax(gate_weights).unsqueeze(-1)
        out = torch.matmul(out, gate_weights).squeeze(-1)
        return out


if __name__ == "__main__":
    # 测试代码
    B = 2
    L = 49
    C = 96
    h = w = 7
    x = torch.randn(B, L, C)
    model = NSA()
    output = model(x, h=h, w=w)
    print(output.shape)