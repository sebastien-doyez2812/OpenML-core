import numpy as np
import torch 

def iou3d(preds, targets):
    num_batches = preds.shape[0]
    num_classes = preds.shape[1]  

    metric_to_return = [0.0] * num_classes
    valid_batches_per_class = [0] * num_classes 

    for idx_current_batch in range(num_batches):
        for idx_class in range(num_classes):
            pred = preds[idx_current_batch, idx_class]
            gt = targets[idx_current_batch, idx_class]

            pred_mask = (pred > 0.5)
            gt_mask = (gt > 0.5)

            intersection = torch.logical_and(pred_mask, gt_mask).sum().item()
            union = torch.logical_or(pred_mask, gt_mask).sum().item()

            if union > 0:
                iou = intersection / union
                metric_to_return[idx_class] += iou
                valid_batches_per_class[idx_class] += 1

    final_metric = []
    for i in range(num_classes):
        if valid_batches_per_class[i] > 0:
            final_metric.append(metric_to_return[i] / valid_batches_per_class[i])
        else:
            final_metric.append(float('nan')) 

    return final_metric

def accuracy3d(preds, targets):
    num_batches = preds.shape[0]
    num_classes = preds.shape[1]  

    metrics_to_return = [0.0] * num_classes
    valid_batches_per_class = [0] * num_classes 

    for idx_current_batch in range(num_batches):
        pred_batch = preds[idx_current_batch] 
        
        if targets.ndim == preds.ndim:
            target_batch = targets[idx_current_batch] # Shape: (C, ...)
        else:
            target_indices = targets[idx_current_batch] # Shape: (...)

        for idx_class in range(num_classes):
            pred = pred_batch[idx_class]
            pred_mask = (pred > 0.5) 

            if targets.ndim == preds.ndim:
                gt_mask = (target_batch[idx_class] > 0.5)
            else:
                gt_mask = (target_indices == idx_class)

            correct = torch.sum(pred_mask == gt_mask).item()
            total = gt_mask.numel()
            
            if total > 0:
                current_acc = correct / total
                metrics_to_return[idx_class] += current_acc
                valid_batches_per_class[idx_class] += 1

    final_metric = []
    for i in range(num_classes):
        if valid_batches_per_class[i] > 0:
            final_metric.append(metrics_to_return[i] / valid_batches_per_class[i])
        else:
            final_metric.append(float('nan')) 

    return final_metric

def Precision3d(preds, labels, nb_classes):
    pred_flatten = preds.flatten()
    label_flatten = labels.flatten()

    precision_tab = []
    for idx_class in range(1, nb_classes+1):
        TP = np.sum((pred_flatten == idx_class) & (label_flatten == idx_class))
        FP = np.sum((pred_flatten == idx_class) & (label_flatten != idx_class))
        if (TP + FP) == 0:
            precision_tab.append(0.0)
        else:
            precision_tab.append(TP/(TP+FP))

    return precision_tab

def Recall3d(preds, labels, nb_classes):
    pred_flatten = preds.flatten()
    label_flatten = labels.flatten()

    precision_tab = []
    for idx_class in range(1, nb_classes+1):
        TP = np.sum((pred_flatten == idx_class) & (label_flatten == idx_class))
        FN = np.sum((pred_flatten != idx_class) & (label_flatten == idx_class))
        if (TP + FN) == 0:
            precision_tab.append(0.0)
        else:
            precision_tab.append(TP/(TP+FN))

    return precision_tab


def F13d(preds, labels, nb_classes):
    current_prec = Precision3d(preds, labels, nb_classes)
    current_recall = Recall3d(preds, labels, nb_classes)

    f1_scores = []
    for idx_class in range(nb_classes):
        if current_prec[idx_class] + current_recall[idx_class] == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.appennd(2* (current_prec[idx_class] * current_recall[idx_class])/ (current_prec[idx_class] + current_recall[idx_class]))

    return f1_scores