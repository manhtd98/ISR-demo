
import torch
from torch import nn
from torch.nn import functional as F


class BALoss(torch.nn.Module):
    def __init__(self):
        super(BALoss, self).__init__()

    def forward(self, output, target):
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        b, c, w, h = output.shape
        sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
        sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
        sobel_x = sobel_x.type_as(output)
        sobel_y = sobel_y.type_as(output)
        weight_x = torch.nn.Parameter(data=sobel_x, requires_grad=False)
        weight_y = torch.nn.Parameter(data=sobel_y, requires_grad=False)
        Ix1 = F.conv2d(output, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(output, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(target, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        #     loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = dx * dy * torch.abs(target - output)
        return torch.mean(loss)

def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

class CharbonnierLoss(torch.nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_func = nn.MSELoss(reduction='mean')
            elif loss_type == 'L1':
                loss_func = nn.L1Loss(reduction='mean')
            elif loss_type == 'Huber':
                loss_func = nn.HuberLoss(reduction='mean')
            elif loss_type == 'CharbonnierLoss':
                loss_func = L1_Charbonnier_loss()
            elif loss_type == 'SmoothL1':
                loss_func = nn.SmoothL1Loss(reduction='mean')
            else:
                raise NotImplementedError
            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_func})

        for l in self.loss:
            if args.local_rank == 0:
                print('Loss Function: {:.3f} * {}'.format(l['weight'], l['type']))
            self.loss_module.append(l['function'])

        device = torch.device(args.device)
        self.loss_module.to(device)

    def forward(self, out, gt, mask=None):
        losses = []

        for i, l in enumerate(self.loss):
            if mask is None:
                loss = l['function'](out, gt)
            else:
                loss = l['function'](out[mask == 1.], gt[mask == 1.])
            effective_loss = l['weight'] * loss
            losses.append(effective_loss)
        return sum(losses)

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss