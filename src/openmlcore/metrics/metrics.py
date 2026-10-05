###             GUIDELINES            ###

# The metrics are define by functions.
# The function should ALWAYS return a list with the value of the metric of each class.
# For example IoU (Intersection of Union) return a list.
# Let's say we have 3 classes:
# Class 0: sky
# Class 1: Road
# Class 2: Car
# We should return a list of 3 elements like [IoU_sky, IoU_Road, IoU_Car]
import torch

def accuracy(y_pred, y_true):
    num_batches = y_true.shape[0]
    num_classes = y_true.shape[1]

    metric_to_return = [0.0] * num_classes
    for idx_current_batch in range(num_batches):
        current_metrics = []
        for idx_class in range(num_classes):
            pred = (y_pred[idx_current_batch, idx_class, :, :] > 0.0).float()
            gt   = y_true[idx_current_batch, idx_class, :, :]

            correct = (pred == gt).float()
            accuracy = (pred == gt).float().mean().item()

            current_metrics.append(accuracy)
        
        for i in range(num_classes):
            metric_to_return[i] += current_metrics[i]

    final_metric = [0.0] * num_classes
    for idx, metric in enumerate(metric_to_return):
        final_metric[idx] = metric / num_batches 
    return final_metric
    
def iou(y_pred, y_true,smooth = 1e-6):
    preds = (y_pred > 0.0).float()
    preds = preds.view(preds.size(0), preds.size(1), -1)
    y_true = y_true.view(y_true.size(0), y_true.size(1), -1)

    intersection = (preds * y_true).sum(dim=2)
    total = preds.sum(dim=2) + y_true.sum(dim=2)
    union = total - intersection

    iou_per_class = (intersection + smooth) / (union + smooth)
    mean_iou_per_class = iou_per_class.mean(dim=0)
    
    return mean_iou_per_class.tolist()


def F1(y_pred, y_true, eps = 1e-7):
    # F1 = 2. precision.recall/(precision + recall)
    recall_res    = recall(y_pred=y_pred, y_true=y_true)
    precision_res = precision(y_pred=y_pred, y_true=y_true)

    F1_values = []
    for idx in range(len(recall_res)):
        current_F1 = 2 * recall_res[idx] * precision_res[idx] / (precision_res[idx] + recall_res[idx] + eps)
        F1_values.append(current_F1)
    return F1_values

def recall(y_pred, y_true, threshold = 0.5, eps = 1e-7):
    # Recall = TP/(TP+FN)
    y_pred_bin = (y_pred >threshold).float()
    y_true     = y_true.float()
    # (0, 2, 3) means we are going to add the batch, H and W, we only keep the class
    TP = torch.sum(y_pred_bin * y_true      , dim= (0, 2, 3)) 
    FN = torch.sum(y_true * (1 - y_pred_bin), dim= (0, 2, 3))
    precision = TP/(TP + FN + eps)
    return precision.tolist()

def precision(y_pred, y_true, threshold = 0.5, eps = 1e-7):
    # Prec = TP/(TP+FP)
    y_pred_bin = (y_pred >threshold).float()
    y_true     = y_true.float()
    # (0, 2, 3) means we are going to add the batch, H and W, we only keep the class
    TP = torch.sum(y_pred_bin * y_true      , dim= (0, 2, 3)) 
    FP = torch.sum((1 - y_true) * y_pred_bin, dim= (0, 2, 3))
    precision = TP/(TP + FP + eps)
    return precision.tolist()