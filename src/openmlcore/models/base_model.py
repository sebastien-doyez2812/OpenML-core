import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

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
        
    def train(self, train_loader: DataLoader,val_loader:DataLoader | None = None, epochs: int = 100, save_checkpoint = True):
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

                dict_metrics_sum = {}
                
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

                        for name_metric, metric_fn in self.metrics.items():
                            metric_value = metric_fn(prediction, Y_batch)

                            if isinstance(metric_value, torch.Tensor):
                                metric_value = metric_value.detach().cpu()
                                if metric_value.ndim > 1:
                                    metric_value = metric_value.mean(dim=0)
                                
                                if metric_value.ndim == 0:
                                    metric_value = metric_value.item()
                                else:
                                    metric_value = metric_value.tolist()

                            elif isinstance(metric_value, np.ndarray):
                                if metric_value.ndim > 1:
                                    metric_value = metric_value.mean(axis=0)
                                
                                if metric_value.ndim == 0:
                                    metric_value = metric_value.item()
                                else:
                                    metric_value = metric_value.tolist()
                            if isinstance(metric_value, list) and len(metric_value) > 0 and isinstance(metric_value[0], list):
                                metric_value = [sum(col) / len(col) for col in zip(*metric_value)]

                            if name_metric not in dict_metrics_sum:
                                if isinstance(metric_value, list):
                                    dict_metrics_sum[name_metric] = [0.0] * len(metric_value)
                                else:
                                    dict_metrics_sum[name_metric] = 0.0

                            if isinstance(metric_value, list):
                                for cls_i, val in enumerate(metric_value):
                                    dict_metrics_sum[name_metric][cls_i] += float(val)
                            else:
                                dict_metrics_sum[name_metric] += float(metric_value)

                num_batchs = len(val_loader)
                metrics_summary_str = []

                for name_metric, val_sum in dict_metrics_sum.items():
                    if isinstance(val_sum, list):
                        avg_list = [v/num_batchs for v in val_sum]
                        mean_all_classes = sum(avg_list)/len(avg_list)

                        class_details = ", ".join([f"C{i}:{v:.3f}" for i, v in enumerate(avg_list)])
                        metrics_summary_str.append(f"{name_metric}_Mean: {mean_all_classes:.4f} ({class_details})")
                    else:
                        avg_val = val_sum / num_batchs
                        metrics_summary_str.append(f"{name_metric}: {avg_val:.4f}")

                formatted_metrics = " | ".join(metrics_summary_str)
                tqdm.write(f"Epoch {current_epoch + 1}/{epochs} — Validation: {formatted_metrics}")
            
            if save_checkpoint:
                self.save_model(f"checkpoint_{self.name}.pt")

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
        dict_metrics_sum = {}

        with torch.no_grad():
            for X_batch, Y_batch in eval_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                predictions = self.model(X_batch)

                for name_metric, metric_formula in self.metrics.items():
                    metric_value = metric_formula(predictions, Y_batch) 
                    
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.cpu().tolist()
                    elif isinstance(metric_value, np.ndarray):
                        metric_value = metric_value.tolist()

                    if name_metric not in dict_metrics_sum:
                        if isinstance(metric_value, list):
                            dict_metrics_sum[name_metric] = [0.0] * len(metric_value)
                        else:
                            dict_metrics_sum[name_metric] = 0.0

                    if isinstance(metric_value, list):
                        for cls_i, val in enumerate(metric_value):
                            dict_metrics_sum[name_metric][cls_i] += val
                    else:
                        dict_metrics_sum[name_metric] += metric_value
            
            num_batches = len(eval_loader)
            final_metrics = {}

            for name_metrics, val_sum in dict_metrics_sum.items():
                if isinstance(val_sum, list):
                    final_metrics[name_metric] = [v / num_batches for v in val_sum]
                else:
                    final_metrics[name_metric] = val_sum / num_batches

        return final_metrics

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

        # To make sure the modele is on the correct device:
        self.model.to(self.device)
        print(f"Model loaded from {file_path}, model is in {self.device} mode.")

    def save_model_in_onnx(self, file_path, input_sample):
        self.model.eval()
        input_sample = input_sample.to(self.device)
        torch.onnx.export(self.model, input_sample, file_path)
        print(f"Model saved in ONNX format to {file_path}")
