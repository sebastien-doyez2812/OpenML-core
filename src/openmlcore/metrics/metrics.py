###             GUIDELINES            ###

# The metrics are define by functions.
# The function should ALWAYS return a list with the value of the metric of each class.
# For example IoU (Intersection of Union) return a list.
# Let's say we have 3 classes:
# Class 0: sky
# Class 1: Road
# Class 2: Car
# We should return a list of 3 elements like [IoU_sky, IoU_Road, IoU_Car]


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
