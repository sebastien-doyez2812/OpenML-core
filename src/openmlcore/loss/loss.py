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
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

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
    