import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmcv.cnn.bricks.transformer import FFN

@MODELS.register_module()
class SparseMoE(BaseModule):
    """稀疏混合专家（Sparse MoE）模块，适用于 Transformer 层。

    Args:
        embed_dims (int): 输入特征的维度。
        feedforward_channels (int): 专家网络的隐藏层维度。
        drop_rate (float): Dropout 概率。
        drop_path_rate (float): DropPath 概率。
        act_cfg (dict): 激活函数配置。
        num_experts (int): 专家数量。
        topk (int): 每个输入选择的专家数量（默认为2）。
        gate_type (str): 门控网络类型（默认为 'linear'）。
        norm_cfg (dict): 归一化层配置。
        init_cfg (dict): 初始化配置。
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate,
                 drop_path_rate,
                 num_shared_experts=4,
                 act_cfg=dict(type='GELU'),
                 num_experts=4,
                 topk=2,
                 gate_type='linear',
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(SparseMoE, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_experts = num_experts
        self.topk = topk
        self.gate_type = gate_type
        self.drop_path_rate = drop_path_rate
        self.dropout = nn.Dropout(drop_rate)

        # 门控网络
        if gate_type == 'linear':
            self.gate = nn.Linear(embed_dims, num_experts)
        else:
            raise NotImplementedError(f'Unsupported gate type: {gate_type}')

        # 初始化专家网络（每个专家是一个独立的FFN）
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=2,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                add_identity=False,  # 禁用残差连接，由外层统一处理
                init_cfg=None)
            self.experts.append(expert)
        self.shared_experts = nn.ModuleList()
        for _ in range(num_shared_experts):
            expert = FFN(embed_dims=embed_dims,
                         feedforward_channels=feedforward_channels,
                         num_fcs=2,
                         ffn_drop=drop_rate,
                         dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                         act_cfg=act_cfg,
                         add_identity=True,
                         init_cfg=None)
            self.shared_experts.append(expert)
        # 归一化层
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x):
        B, N, C = x.shape  # Batch, Tokens, Channels
        x_flat = x.view(-1, C)  # 展平为 (B*N, C)

        # 门控网络计算专家权重
        gate_logits = self.gate(x_flat)  # (B*N, num_experts)
        gates, indices = torch.topk(gate_logits, self.topk, dim=1)  # 取topk
        gates = torch.softmax(gates, dim=1)  # 归一化权重

        # 初始化输出
        output = torch.zeros_like(x_flat)

        # 将输入复制topk次以匹配路由
        x_repeated = x_flat.repeat_interleave(self.topk, dim=0)  # (B*N*topk, C)
        gates_flat = gates.view(-1)  # (B*N*topk)
        indices_flat = indices.view(-1)  # (B*N*topk)

        # 遍历每个专家，处理分配到的输入
        for expert_idx in range(self.num_experts):
            # 筛选当前专家处理的输入
            mask = indices_flat == expert_idx
            if not mask.any():
                continue  # 无输入分配到该专家

            expert_input = x_repeated[mask]  # (num_selected, C)
            expert_gates = gates_flat[mask]  # (num_selected)

            # 专家处理（添加虚拟批次维度）
            expert_output = self.experts[expert_idx](expert_input.unsqueeze(0)).squeeze(0)
            expert_output = expert_output * expert_gates.unsqueeze(-1)  # 加权

            # 将输出累加到对应位置
            original_indices = torch.arange(B*N, device=x.device).repeat_interleave(self.topk)
            selected_indices = original_indices[mask]
            output.index_add_(0, selected_indices, expert_output)

        # 恢复形状并归一化
        output = output.view(B, N, C)
        for expert in self.shared_experts:
            output += expert(x)
        
        output = self.norm(output)
        output = self.dropout(output)
        
        return x + output  # 残差连接