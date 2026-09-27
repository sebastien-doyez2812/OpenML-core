import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self, name: str = "BaseLoss", doc_url: str = ""):
        super().__init__()
        self.name = name
        self.doc_url = doc_url

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def documentation(self) -> str:
        print(f"Documentation for {self.name} is available here:\n{self.doc_url}")
        try:
            import webbrowser
            webbrowser.open(self.doc_url)
        except Exception as e:
            print(f"[-] {e}")

#############################
#   Class Loss Functions    #
#############################

class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__(name = "CrossEntropyLoss", doc_url = "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html")
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(outputs, targets)


class MeanSquaredErrorLoss(BaseLoss):
    def __init__(self):
        super().__init__(name = "MeanSquaredErrorLoss", doc_url = "https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html")
        self.loss_fn = nn.MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(outputs, targets)

class DiceLoss(BaseLoss):
    def __init__(self, smooth=1e-6 ):
        super().__init__(name = "DiceLoss", doc_url = "https://arxiv.org/abs/1606.04797")
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = outputs.shape[1]

        if num_classes == 1:
            outputs = torch.sigmoid(outputs)
            # Correction: targets.dim() avec des parenthèses
            target = targets.unsqueeze(1).float() if targets.dim() == 3 else targets.float()
            
            intersection = (outputs * target).sum(dim=(2, 3))
            union = outputs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice_score.mean()

        # 2. CAS MULTI-CLASSES (output_channels > 1)
        else:
            outputs = F.softmax(outputs, dim=1)

            # Priorité explicite des conditions avec parenthèses
            if targets.dim() == 3 or (targets.dim() == 4 and targets.shape[1] == 1):
                targets = targets.squeeze(1) if targets.dim() == 4 else targets
                
                # Correction: conversion explicite en .long() pour F.one_hot
                targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes) # [B, H, W, C]
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()         # [B, C, H, W]
            else:
                targets_one_hot = targets.float()

            intersection = (outputs * targets_one_hot).sum(dim=(2, 3))
            union = outputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            
            return 1.0 - dice_score.mean()

class FocalLoss(BaseLoss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name = "FocalLoss", doc_url = "https://arxiv.org/abs/1708.02002")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class HuberLoss(BaseLoss):
    def __init__(self, delta=1.0):
        super().__init__(name = "HuberLoss", doc_url = "https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html")
        self.delta = delta

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        error = outputs - targets
        is_small_error = torch.abs(error) < self.delta
        small_error_loss = 0.5 * error ** 2
        large_error_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.where(is_small_error, small_error_loss, large_error_loss).mean()

class KLDivergenceLoss(BaseLoss):
    def __init__(self):
        super().__init__(name = "KLDivergenceLoss", doc_url = "https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html")
        self.eps = 1e-8

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = torch.softmax(outputs, dim=1)
        targets = torch.softmax(targets, dim=1)
        kl_div = nn.functional.kl_div(torch.log(outputs + self.eps), targets, reduction='batchmean')
        return kl_div


####################################
#   Custom Loss implementation     #
####################################

class LossFactory(BaseLoss):
    @staticmethod
    def create(name: str, **kwargs ) -> BaseLoss:
        match name:
            case "CrossEntropyLoss":
                return CrossEntropyLoss()
            case "MeanSquaredErrorLoss":
                return MeanSquaredErrorLoss()
            case "DiceLoss":
                return DiceLoss()
            case "FocalLoss":
                return FocalLoss()
            case "HuberLoss":
                return HuberLoss()
            case "KLDivergenceLoss":
                return KLDivergenceLoss()
            case _:
                raise ValueError(f"Unknown loss function: {name}")

class CustomLoss(BaseLoss):
    def __init__(self, loss_fcns: list[BaseLoss], name: str, coefficients: list[float]):
        assert len(loss_fcns) == len(coefficients), "The number of loss functions must match the number of coefficients."
        super().__init__(name = name, doc_url = None)
        self.loss_fcns = loss_fcns
        self.coefficients = coefficients

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss_fn, coef in zip(self.loss_fcns, self.coefficients):
            total_loss += coef * loss_fn(outputs, targets)
        return total_loss
    