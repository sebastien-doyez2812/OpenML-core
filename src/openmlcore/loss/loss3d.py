import torch
import torch.nn as nn
import torch.nn.functional as F

class Base3dLoss(nn.Module):
    def __init__(self, name: str= "BaseLoss3d", doc_url:str= ""):
        super().__init__()
        self.name = name
        self.type = "segmentation3d"
        self.smooth = 1e-6
        self.doc_url = doc_url

    def get_type(self):
        return self.type

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def documentation(self) -> str:
        print(f"Documentation available here:{self.doc_url}")
        try:
            import webbrowser
            webbrowser.open(self.doc_url)
        except Exception as e:
            print(f"[-] {e}")


class CustomLoss3d(Base3dLoss):
    def __init__(self, loss_fcns: list[Base3dLoss], name: str, coefficients: list[float]):
        assert len(loss_fcns) == len(coefficients), "The number of loss functions must match the number of coefficients."
        super().__init__(name=name, doc_url=None)
        self.loss_fcns = nn.ModuleList(loss_fcns)
        self.coefficients = coefficients

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, coef in zip(self.loss_fcns, self.coefficients):
            total_loss += coef * loss_fn(outputs, targets)
        return total_loss




class CrossEntropyLoss(Base3dLoss):
    def __init__(self, name = "CrossEntropy", weights = None, doc_url = ""):
        super().__init__(name, doc_url)
        self.weights = weights

        device = "cuda" if torch.cuda.is_available() else "cpu"
    
        pos_weights = None
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                pos_weights = torch.tensor(weights, device=device, dtype=torch.float32)
            else:
                pos_weights = weights.to(device)

        self.loss_fn = nn.CrossEntropyLoss(weight=pos_weights)
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_fn(outputs, targets.long())

class DiceLoss3d(Base3dLoss):
    def __init__(self, weights = None):
        super().__init__(name="DiceLoss3d")
        if weights is None:
            raise ValueError("weights need to be set!")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not isinstance(weights, torch.Tensor):
            self.weights = torch.tensor(weights, device=device, dtype=torch.float32)
        else:
            self.weights = weights.to(device)

    def forward(self, outputs, targets):
        probs = F.softmax(outputs, dim=1)
        B, C, N = probs.shape

        if targets.min() < -1 or targets.max() > C:
            raise ValueError("target at -1")

        targets_one_hot = F.one_hot(targets.long(), num_classes=C)
        targets_one_hot = targets_one_hot.permute(0, 2, 1).float()

        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2))
        union = torch.sum(probs + targets_one_hot, dim=(0, 2))

        dice_score = (2.0 * intersection) / (self.smooth + union)

        dice_loss = 0
        for idx_class, weight in enumerate(self.weights):
            dice_loss += weight * (1 - dice_score[idx_class])

        return dice_loss
