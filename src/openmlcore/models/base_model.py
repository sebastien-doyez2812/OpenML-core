import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseModel(nn.Module):
    def __init__(self, name: str = "BaseModel", model = None, metrics = None, loss_fn = None, optimizer = None):
        super().__init__()
        self.name = name
        self.model = model
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def help(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def create_model(self):
        raise NotImplementedError("Subclasses should implement this method.")
        

    def train(self, train_loader: DataLoader, epochs: int = 100):
        if self.name == "BaseModel":
            raise NotImplementedError("Subclasses should implement this method.")
        if self.model == None or self.name == "BaseModel":
            raise ValueError("Model has not been created. Please call create_model() before training.")
        if self.loss_fn == None:
            raise ValueError("Loss function has not been set. Please set a loss function before training.")
        if self.optimizer == None:
            raise ValueError("Optimizer has not been set. Please set an optimizer before training.")
        
        self.model.train()
        for current_epoch in range(epochs):
            total_loss = 0.0
            print(f"Epoch {current_epoch + 1}/{epochs}")
            for X_batch, Y_batch in train_loader:
                self.optimizer.zero_grad()
                prediction = self.model(X_batch)
                loss = self.loss_fn(prediction, Y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Loss = {avg_loss:.4f}")
            
    def predict(self, data):
        if self.model == None or self.name == "BaseModel":
            raise ValueError("Model has not been created. Please call create_model() before predicting.")

        self.model.eval()
        with torch.no_grad():
            return self.model(data)

    def evaluate(self, eval_loader: DataLoader):
        if self.model == None or self.name == "BaseModel":
            raise ValueError("Model has not been created. Please call create_model() before evaluating.")
        if self.metrics is None:
            raise ValueError("No metrics have been defined for evaluation. Please define metrics before evaluating.")
        
        self.model.eval()
        dict_metrics = {}

        with torch.no_grad():
            for X_batch, Y_batch in eval_loader:
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
        print(f"This function will return information about the {self.name} model.")
        pass

    def save_model(self, file_path):
        self.model.eval()
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, is_train_mode=False):
        self.model.load_state_dict(torch.load(file_path, weights_only=True))
        if is_train_mode:
            self.model.train()
        else:
            self.model.eval()
        print(f"Model loaded from {file_path}")

    def save_model_in_onnx(self, file_path, input_sample):
        self.model.eval()
        torch.onnx.export(self.model, input_sample, file_path)
        print(f"Model saved in ONNX format to {file_path}")
