# mmdet/models/losses/load_balance_loss.py

import torch
import torch.nn as nn
from mmengine.model import BaseModule


class LoadBalanceLoss(BaseModule):
    def __init__(self,
                 loss_weight=0.1,
                 aux_loss_factor=0.05,
                 eps=1e-6,
                 reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.aux_loss_factor = aux_loss_factor
        self.eps = eps
        self.reduction = reduction

    def forward(self, gate_logits, expert_mask):
        """
        Args:
            gate_logits (Tensor): shape [num_groups, num_experts]
            expert_mask (Tensor): shape [num_groups, num_experts]
        Returns:
            Tensor: scalar
        """
        # 计算每个专家的路由数量
        route_prob = torch.softmax(gate_logits, dim=-1)  # [num_groups, num_experts]
        expert_capacity = expert_mask.sum(dim=0) + self.eps  # [num_experts]

        # 负载均衡公式：(E[route_prob] - Var(route_prob))
        mean_route = route_prob.mean(dim=0)
        var_route = route_prob.var(dim=0)

        load_balance = (mean_route * var_route).sum()
        loss = self.loss_weight * load_balance

        return loss