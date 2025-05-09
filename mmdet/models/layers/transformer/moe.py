import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule, ModuleList
from mmdet.registry import MODELS
from mmcv.cnn.bricks.transformer import FFN, build_dropout

@MODELS.register_module()
class MoE(BaseModule):
    """Mixture of Experts (MoE) 模块，适用于 DINO 模型的 Transformer 层。

    Args:
        in_channels (int): 输入特征的通道数。
        hidden_channels (int): 专家网络的隐藏层通道数。
        num_experts (int): 专家网络的数量。
        gate_type (str): 门控网络类型（默认为 'linear'）。
        norm_cfg (dict): 归一化层配置（默认为 LN）。
        init_cfg (dict): 初始化配置（可选）。
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate,
                 drop_path_rate,
                 num_shared_experts=2,
                 act_cfg=dict(type='GELU'),
                 num_experts=4,
                 gate_type='linear',
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(MoE, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_experts = num_experts
        self.gate_type = gate_type
        self.drop_path_rate = drop_path_rate
        self.dropout = nn.Dropout(drop_rate)

        # 门控网络（决定输入分配到哪个专家）
        if gate_type == 'linear':
            self.gate = nn.Linear(embed_dims, num_experts+num_shared_experts)
        else:
            raise NotImplementedError(f'Gate type {gate_type} not supported')

        # 专家网络（每个专家是一个全连接层）
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
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = FFN(embed_dims=embed_dims,
                        feedforward_channels=feedforward_channels,
                        num_fcs=2,
                        ffn_drop=drop_rate,
                        dropout_layer=dict(type='DropPath', drop_prob=self.drop_path_rate),
                        act_cfg=act_cfg,
                        add_identity=True,
                        init_cfg=None)
            self.experts.append(expert)

        # 归一化层（可选）
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x):
        """前向传播：
        1. 门控网络计算权重
        2. 输入分配到所有专家
        3. 加权聚合专家输出
        """
        B, N, C = x.shape  # B: batch_size, N: tokens, C: channels

        # 门控网络输出权重
        gate_weights = self.gate(x)  # (B, N, num_experts)
        gate_weights = gate_weights.softmax(dim=-1)  # 归一化权重

        # 专家网络处理输入
        expert_outputs = []
        for expert in self.experts:
            # 每个专家处理所有 tokens
            out = expert(x)  # (B, N, C)
            expert_outputs.append(out.unsqueeze(-2))  # (B, N, 1, C)
        for expert in self.shared_experts:
            # 每个专家处理所有 tokens
            out = expert(x)  # (B, N, C)
            expert_outputs.append(out.unsqueeze(-2))  # (B, N, 1, C)
        # 将专家输出堆叠为 (B, N, num_experts, C)
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        expert_outputs = expert_outputs.permute(0, 1, 3, 2) # (B, N, C, num_experts)
        # 加权求和：每个专家的输出乘以对应的权重
        weighted_sum = torch.matmul(expert_outputs , gate_weights.unsqueeze(-1)).squeeze(-1)
        weighted_sum = self.dropout(weighted_sum)
        weighted_sum = self.norm(weighted_sum)

        

        return x + weighted_sum  # 残差连接