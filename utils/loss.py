import torch
import torch.nn as nn

class CustomDepthLoss(nn.Module):
    def __init__(self, edge_weight=0.1):
        super().__init__()
        self.l1_criterion = nn.L1Loss()
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        l1_loss = self.l1_criterion(pred, target)
        
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_loss_x = self.l1_criterion(pred_dx, target_dx)
        grad_loss_y = self.l1_criterion(pred_dy, target_dy)
        
        total_loss = l1_loss + self.edge_weight * (grad_loss_x + grad_loss_y)
        
        return total_loss