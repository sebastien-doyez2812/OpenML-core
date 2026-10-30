import numpy as np

def iou3d(preds, labels, nb_classes):
    pred_flatten = preds.flatten()
    label_flatten = labels.flatten()

    iou_tab = []
    for idx_class in range(1, nb_classes+1):
        pred_mask = (pred_flatten == idx_class)
        target_mask = (label_flatten == idx_class)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            iou_tab.append()
        else:
            iou_tab.append(intersection / union)

    return iou_tab

def accuracy3d(preds, labels):
    pred_flatten = preds.flatten()
    label_flatten = labels.flatten()
    
    correct = np.sum(pred_flatten == label_flatten)
    total = len(label_flatten)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


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