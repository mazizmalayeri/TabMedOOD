import torch
from models.predictive_models import apply_model
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.special import softmax

@torch.no_grad()
def evaluate_predictions(model, X, y, part, batch_size, device):
    
    """
    Evaluate the classification model's predictions on the val/test data.

    Parameters:
    -----------
    model: torch.nn.Module
        The trained machine learning model to evaluate.

    X: dict of torch.Tensor
        Dictionary containing data splits ('train', 'val', 'test') as keys and torch.Tensor objects
        as values for features.

    y: dict of torch.Tensor
        Dictionary containing data splits ('train', 'val', 'test') as keys and torch.Tensor objects
        as values for ground truth labels.

    part: str
        The data split to evaluate ('train', 'val', 'test').

    batch_size : int
        Batch size to use during evaluation.

    device: torch.device
        The device (CPU or GPU) to perform evaluation on.

    Returns:
    --------
    accuracy: float
        Accuracy of the model's predictions.
    f_score: float
        F1 score of the model's predictions.
    auc_preds: float
        Area Under the Curve (AUC) for binary predictions based on model's class predictions.
    auc_scores : float
        Area Under the Curve (AUC) based on probability assigned to classes.
    """
    
    model.eval()
    prediction_logits = []
    for idx in range(len(X[part]) // batch_size):
        batch = X[part][idx*batch_size:(idx+1)*batch_size].to(device)
        prediction_logits.append(apply_model(model, batch))
    prediction_logits = torch.cat(prediction_logits).cpu().numpy()
    target = y[part][:(idx+1)*batch_size].cpu().numpy()

    prediction_labels = prediction_logits.argmax(1)
    
    accuracy = accuracy_score(target, prediction_labels)
    f_score = f1_score(target, prediction_labels)
    auc_preds = roc_auc_score(target, prediction_labels)
    
    prediction_scores = softmax(prediction_logits, axis=1)[:,1]
    auc_scores = roc_auc_score(target, prediction_scores)

    return accuracy, f_score, auc_preds, auc_scores
