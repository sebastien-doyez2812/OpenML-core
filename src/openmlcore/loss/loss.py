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

    def _to_one_hot_clean(self, targets: torch.Tensor, num_classes: int, index_ignore: int):
        """Helper pour transformer [B, H, W] en [B, C, H, W] One-Hot tout en isolant 255."""
        valids = (targets != index_ignore)
        mask_cleaned = targets.clone()
        mask_cleaned[~valids] = 0
        
        targets_one_hot = F.one_hot(mask_cleaned.long(), num_classes=num_classes) # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()             # [B, C, H, W]
        
        valid_mask = valids.unsqueeze(1).float() # [B, 1, H, W]
        return targets_one_hot * valid_mask, valid_mask

#############################
#   Class Loss Functions    #
#############################

class CrossEntropyLoss(BaseLoss):
    def __init__(self, index_ignore=255):
        super().__init__(name="CrossEntropyLoss", doc_url="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html")
        self.index_ignore = index_ignore
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.index_ignore)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        return self.loss_fn(outputs, targets.long())

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

class DiceLoss(BaseLoss):
    def __init__(self, smooth=1e-6, index_ignore=255):
        super().__init__(name="DiceLoss", doc_url="https://arxiv.org/abs/1606.04797")
        self.smooth = smooth
        self.index_ignore = index_ignore

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = outputs.shape[1]
        probs = F.softmax(outputs, dim=1)
        
        targets_one_hot, valid_mask = self._to_one_hot_clean(targets, num_classes, self.index_ignore)
        probs = probs * valid_mask

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            
        return 1.0 - dice_score.mean()


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
        super().__init__(name=name, doc_url=None)
        self.loss_fcns = loss_fcns
        self.coefficients = coefficients

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, coef in zip(self.loss_fcns, self.coefficients):
            total_loss += coef * loss_fn(outputs, targets)
        return total_loss
    