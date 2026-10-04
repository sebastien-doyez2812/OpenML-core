import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseLoss(nn.Module):
    def __init__(self, name: str = "BaseLoss", doc_url: str = ""):
        super().__init__()
        self.name = name
        self.smooth = 1e-6
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

class BCEWithLogitsLoss(BaseLoss):
    def __init__(self, weights=None):
        super().__init__(name="BCEWithLogitsLoss", doc_url="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pos_weights = None
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                pos_weights = torch.tensor(weights, device=device, dtype=torch.float32)
            else:
                pos_weights = weights.to(device)

            if pos_weights.ndim == 1:
                pos_weights = pos_weights.view(-1, 1, 1)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(outputs, targets.float())

class DiceLoss(BaseLoss):
    def __init__(self, weights = None):
        super().__init__(name="DiceLoss", doc_url="https://arxiv.org/abs/1606.04797")
        self.weights = weights

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(outputs)
        probs   = probs.view  (probs.size(0)  , probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        intersection = (probs * targets).sum(dim=2)
        union        = probs.sum(dim=2) + targets.sum(dim=2) 

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.weights is not None:
            if not isinstance(self.weights, torch.Tensor):
                weights_tensor = torch.tensor(self.weights, device=outputs.device, dtype=outputs.dtype)
            else:
                weights_tensor = self.weights.to(outputs.device)

            dice_loss = dice_loss * weights_tensor.unsqueeze(0)

            return dice_loss.sum(dim=1).mean()
        return dice_loss.mean()
    
# TODO:
# 1 Hot encoding loss
# 


class MeanSquaredErrorLoss(BaseLoss):
    def __init__(self, index_ignore=255):
        super().__init__(name="MeanSquaredErrorLoss", doc_url="https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html")
        self.index_ignore = index_ignore

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = outputs.shape[1]
        probs = F.softmax(outputs, dim=1)
        targets_one_hot, valid_mask = self._to_one_hot_clean(targets, num_classes, self.index_ignore)
        
        mse = (probs - targets_one_hot) ** 2
        mse = mse * valid_mask
        
        num_valids = valid_mask.sum() * num_classes
        return mse.sum() / torch.clamp(num_valids, min=1.0)



class FocalLoss(BaseLoss):
    def __init__(self, alpha=0.25, gamma=2.0, index_ignore=255):
        super().__init__(name="FocalLoss", doc_url="https://arxiv.org/abs/1708.02002")
        self.alpha = alpha
        self.gamma = gamma
        self.index_ignore = index_ignore

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = outputs.shape[1]
        targets_one_hot, valid_mask = self._to_one_hot_clean(targets, num_classes, self.index_ignore)
        
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        focal_loss = focal_loss * valid_mask
        num_valids = valid_mask.sum() * num_classes
        return focal_loss.sum() / torch.clamp(num_valids, min=1.0)
    
class MeanSquaredErrorLoss(BaseLoss):
    def __init__(self, index_ignore=255):
        super().__init__(name="MeanSquaredErrorLoss", doc_url="https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html")
        self.index_ignore = index_ignore

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = outputs.shape[1]
        probs = F.softmax(outputs, dim=1)
        targets_one_hot, valid_mask = self._to_one_hot_clean(targets, num_classes, self.index_ignore)
        
        mse = (probs - targets_one_hot) ** 2
        mse = mse * valid_mask
        
        num_valids = valid_mask.sum() * num_classes
        return mse.sum() / torch.clamp(num_valids, min=1.0)

class HuberLoss(BaseLoss):
    def __init__(self, delta=1.0, index_ignore=255):
        super().__init__(name="HuberLoss", doc_url="https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html")
        self.delta = delta
        self.index_ignore = index_ignore

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = outputs.shape[1]
        probs = F.softmax(outputs, dim=1)
        targets_one_hot, valid_mask = self._to_one_hot_clean(targets, num_classes, self.index_ignore)
        
        error = probs - targets_one_hot
        abs_error = torch.abs(error)
        
        huber = torch.where(
            abs_error < self.delta,
            0.5 * (error ** 2),
            self.delta * (abs_error - 0.5 * self.delta)
        )
        huber = huber * valid_mask
        
        num_valids = valid_mask.sum() * num_classes
        return huber.sum() / torch.clamp(num_valids, min=1.0)
    
class KLDivergenceLoss(BaseLoss):
    def __init__(self, index_ignore=255):
        super().__init__(name="KLDivergenceLoss", doc_url="https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html")
        self.eps = 1e-8
        self.index_ignore = index_ignore

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = outputs.shape[1]
        log_probs = F.log_softmax(outputs, dim=1)
        targets_one_hot, valid_mask = self._to_one_hot_clean(targets, num_classes, self.index_ignore)
        
        # KL-Div = p(x) * (log(p(x)) - log(q(x)))
        # Pour une distribution One-Hot, cela se réduit à : - log(q(x)) sur la classe vraie
        kl = F.kl_div(log_probs, targets_one_hot, reduction='none')
        kl = kl * valid_mask
        
        num_valids = valid_mask.sum() * num_classes
        return kl.sum() / torch.clamp(num_valids, min=1.0)

####################################
#   Custom Loss implementation     #
####################################

class LossFactory:
    @staticmethod
    def create(name: str, **kwargs ) -> BaseLoss:
        match name:
            case "BCE":
                return BCEWithLogitsLoss(**kwargs)
            case "MeanSquaredErrorLoss":
                return MeanSquaredErrorLoss(**kwargs)
            case "DiceLoss":
                return DiceLoss(**kwargs)
            case "FocalLoss":
                return FocalLoss(**kwargs)
            case "HuberLoss":
                return HuberLoss(**kwargs)
            case "KLDivergenceLoss":
                return KLDivergenceLoss(**kwargs)
            case _:
                raise ValueError(f"Unknown loss function: {name}")

class CustomLoss(BaseLoss):
    def __init__(self, loss_fcns: list[BaseLoss], name: str, coefficients: list[float]):
        assert len(loss_fcns) == len(coefficients), "The number of loss functions must match the number of coefficients."
        super().__init__(name=name, doc_url=None)
        self.loss_fcns = nn.ModuleList(loss_fcns)
        self.coefficients = coefficients

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, coef in zip(self.loss_fcns, self.coefficients):
            total_loss += coef * loss_fn(outputs, targets)
        return total_loss
    