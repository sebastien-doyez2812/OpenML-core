import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class BaseModel(nn.Module):
    def __init__(self, name: str = "BaseModel", model = None, metrics = None, loss_fn = None, optimizer = None, device = None):
        super().__init__()
        self.name = name
        self.model = model
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def help(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def create_model(self):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def train(self, train_loader: DataLoader,val_loader:DataLoader | None = None, epochs: int = 100):
        if self.name == "BaseModel":
            raise NotImplementedError("Subclasses should implement this method.")
        if self.model == None or self.name == "BaseModel":
            raise ValueError("Model has not been created. Please call create_model() before training.")
        if self.loss_fn == None:
            raise ValueError("Loss function has not been set. Please set a loss function before training.")
        if self.optimizer == None:
            raise ValueError("Optimizer has not been set. Please set an optimizer before training.")

        self.model.to(self.device)
        for current_epoch in range(epochs):
            # For each loop: a first phase of training and then validation.
            # Training:
            self.model.train()
            total_loss = 0.0
            total_steps = len(train_loader)
            progress_bar = tqdm(
                train_loader,
                total=total_steps,
                desc=f"Epoch {current_epoch + 1}/{epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
            for idx, (X_batch, Y_batch) in enumerate(progress_bar, start=1):
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                prediction = self.model(X_batch)
                loss = self.loss_fn(prediction, Y_batch)
                loss.backward()
                self.optimizer.step()
                batch_loss = loss.item()
                total_loss += batch_loss
                progress_bar.set_postfix(
                    batch_loss=f"{batch_loss:.4f}",
                    avg_loss=f"{total_loss / idx:.4f}",
                )
            avg_loss = total_loss / len(train_loader)
            tqdm.write(f"Epoch {current_epoch + 1}/{epochs} — average loss: {avg_loss:.4f}")

            # Validation:
            if val_loader:
                self.model.eval()
                dict_metrics = {name:0.0 for name, _ in self.metrics}
                val_progress_bar = tqdm(
                    val_loader,
                    total=len(val_loader),
                    desc=f"Epoch {current_epoch + 1}/{epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                )
                    # Take a portion of the data and test:
                with torch.no_grad():
                    for idx, (X_batch, Y_batch) in enumerate(val_progress_bar, start=1):
                        X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                        prediction = self.model(X_batch)
                        for name_metric, metric_formula in self.metrics:
                            metric_value = metric_formula(prediction, Y_batch) 
                            if name_metric not in dict_metrics:
                                dict_metrics[name_metric] = 0.0
                            dict_metrics[name_metric] += metric_value.item()
                        
                    for name_metric in dict_metrics:
                        dict_metrics[name_metric] /= len(val_loader)
                    # tqdm.write(f"Epoch {current_epoch + 1}/{epochs} — " + f"{name_metric} = {dict_metrics[name_metric]}" for name_metric, _ in self.metrics)
                    metrics_str = " | ".join([f"{name}: {dict_metrics[name]:.4f}" for name, _ in self.metrics])
                    tqdm.write(f"Epoch {current_epoch + 1}/{epochs} — Validation: {metrics_str}")               

            
    def predict(self, data):
        if self.model == None or self.name == "BaseModel":
            raise ValueError("Model has not been created. Please call create_model() before predicting.")

        if data is not None:
            data = data.to(self.device)
            self.model.eval()
            with torch.no_grad():
                return self.model(data)
        else:
            raise ValueError("Data is None!")
        

    def evaluate(self, eval_loader: DataLoader):
        if self.model == None or self.name == "BaseModel":
            raise ValueError("Model has not been created. Please call create_model() before evaluating.")
        if self.metrics is None:
            raise ValueError("No metrics have been defined for evaluation. Please define metrics before evaluating.")

        self.model.eval()
        dict_metrics = {}

        with torch.no_grad():
            for X_batch, Y_batch in eval_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                predictions = self.model(X_batch)

                for name_metric, metric_formula in self.metrics:
                    metric_value = metric_formula(predictions, Y_batch) 
                    if name_metric not in dict_metrics:
                        dict_metrics[name_metric] = 0.0
                    dict_metrics[name_metric] += metric_value.item()

            for name_metric in dict_metrics:
                dict_metrics[name_metric] /= len(eval_loader)
                
        return dict_metrics

    def get_model_info(self):
        if self.model == None or self.name == "BaseModel":
            raise ValueError("Model has not been created. Please call create_model() before getting model info.")
        
        return summary(self.model)

    def save_model(self, file_path):
        self.model.eval()
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, is_train_mode=False):
        self.model.load_state_dict(torch.load(file_path, weights_only=True, map_location=self.device))
        if is_train_mode:
            self.model.train()
        else:
            self.model.eval()
        print(f"Model loaded from {file_path}")

    def save_model_in_onnx(self, file_path, input_sample):
        self.model.eval()
        input_sample = input_sample.to(self.device)
        torch.onnx.export(self.model, input_sample, file_path)
        print(f"Model saved in ONNX format to {file_path}")
