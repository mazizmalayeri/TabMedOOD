from training.evaluate import evaluate_predictions
from models.predictive_models import apply_model
import torch.utils.data as data_utils

def train_predictive_model(model, optimizer, criterion, X, y, batch_size, n_epochs, device, report_frequency):
    
    """
    Train a predictive model using the specified optimizer and loss criterion.

    Args:
        model (torch.nn.Module): The predictive model to train.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        criterion (torch.nn.Module): The loss criterion for training.
        X (dict): Dictionary containing data splits, e.g., 'train', 'val', and 'test'.
        y (dict): Dictionary containing corresponding labels for data splits.
        batch_size (int): Batch size for training.
        n_epochs (int): Number of training epochs.
        device (torch.device): The device to use for training (e.g., 'cpu' or 'cuda').
        report_frequency (int): Frequency at which to report training loss.
    """
    
    evaluation = evaluate_predictions(model, X, y, 'val', batch_size, device)
    print(f'Test score before training: {evaluation[0]:.4f} {evaluation[1]:.4f} {evaluation[2]:.4f} {evaluation[3]:.4f}')
    
    train = data_utils.TensorDataset(X['train'], y['train'])
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        #for batch_idx in range(len(X['train']) // batch_size):
        
            #x_batch = X['train'][batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
            #y_batch = y['train'][batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            loss = criterion(apply_model(model, x_batch), y_batch)
            loss.backward()
            optimizer.step()
            if batch_idx % report_frequency == 0:
                print(f'(epoch) {epoch} (batch) {batch_idx} (loss) {loss.item():.4f}')

        val_score = evaluate_predictions(model, X, y, 'val', batch_size, device)
        print(f'Epoch {epoch:03d} | Validation Accuracy: {val_score[0]:.4f} | Validation f1 score: {val_score[1]:.4f} | Validation auc (preds): {val_score[2]:.4f} | Validation auc (probs): {val_score[3]:.4f}', end='')
