import torch
import torch.nn as nn


class Base3dLoss(nn.Module):
    def __init__(self, name: str= "BaseLoss3d", doc_url:str= ""):
        super().__init__()
        self.name = name
        self.smooth = 1e-6
        self.doc_url = doc_url

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def documentation(self) -> str:
        print(f"Documentation available here:{self.doc_url}")
        try:
            import webbrowser
            webbrowser.open(self.doc_url)
        except Exception as e:
            print(f"[-] {e}")

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
            if pos_weights.ndim == 1:
                pos_weights = pos_weights.view(-1, 1, 1)

        self.loss_fn = nn.CrossEntropyLoss(weight=pos_weights)
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_fn(outputs, targets.float())