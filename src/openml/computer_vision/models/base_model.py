import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, name:str):
        super(BaseModel, self).__init__(name=name)

    def help(self):
        print("This function will open the documentation for the model.")
        pass
    

    def train(self, data):
        print("This function will train the model with the provided data.")
        pass

    def predict(self, data):
        print("This function will make predictions using the trained model.")
        pass

    def evaluate(self, data):
        print("This function will evaluate the model's performance on the provided data.")
        pass

    def get_model_info(self):
        print("This function will return information about the model.")
        pass

    def save_model(self, file_path):
        print(f"This function will save the model to the specified file path: {file_path}.")
        pass

    def get_metrics(self):
        print("This function will return the evaluation metrics of the model.")
        pass

    def load_board(self):
        pass