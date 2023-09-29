from sklearn.model_selection import train_test_split
import sklearn
import torch
import numpy as np

def data_imbalance(in_features_np, in_label_np):
        
        """
        Address data imbalance by oversampling minority classes.

        Parameters:
        -----------
        in_features_np: np.ndarray
                Input features as a NumPy array.
        in_label_np: np.ndarray
                Input labels as a NumPy array.

        Returns:
        --------
        tuple
                A tuple containing two NumPy arrays:
                - in_features_np_sub: The balanced feature dataset.
                - in_label_np_sub: The corresponding balanced labels.

        """
        p = np.random.permutation(len(in_label_np))
        in_features_np, in_label_np = in_features_np[p], in_label_np[p]
        
        unique_labels = np.unique(in_label_np)
        num_samples = np.zeros(unique_labels.shape)
        for label in unique_labels:
            num_samples[label] = (in_label_np==label).sum()
        
        '''
        to_hold = int(num_samples.min())
        print('Holding', to_hold, 'samples for each class.')
        
        in_features_np_sub = in_features_np[in_label_np==0][:to_hold]
        in_label_np_sub = in_label_np[in_label_np==0][:to_hold]
        for label in unique_labels:
            if label!=0:
              in_features_np_sub = np.concatenate((in_features_np_sub, in_features_np[in_label_np==label][:to_hold]))
              in_label_np_sub = np.concatenate((in_label_np_sub, in_label_np[in_label_np==label][:to_hold]))
        '''
        
        num_repeat = num_samples.max()//num_samples
        
        in_features_np_sub = np.repeat(in_features_np[in_label_np==0], repeats=num_repeat[0], axis=0)
        in_label_np_sub = np.repeat(in_label_np[in_label_np==0], repeats=num_repeat[0], axis=0)
        for label in unique_labels:
            if label!=0:
              in_features_np_sub = np.concatenate((in_features_np_sub, np.repeat(in_features_np[in_label_np==label], repeats=num_repeat[label], axis=0)))
              in_label_np_sub = np.concatenate((in_label_np_sub, np.repeat(in_label_np[in_label_np==label], repeats=num_repeat[label], axis=0)))
        
        p = np.random.permutation(len(in_label_np_sub))
        in_features_np_sub, in_label_np_sub = in_features_np_sub[p], in_label_np_sub[p]
        
        return in_features_np_sub, in_label_np_sub
            
            
            
def split_data(in_features_np, in_label_np, train_size=0.7, random_state=0, handle_imbalance_data=False):
    """
    Split input data into training, validation, and test sets.

    Parameters:
    -----------
    in_features_np: np.ndarray
        Input features as a NumPy array.
    in_label_np: np.ndarray
        Input labels as a NumPy array.
    train_size: float, optional (default=0.7)
        Proportion of the data to include in the training set.
    random_state: int, optional (default=0)
        Seed used by the random number generator for reproducibility.
    handle_imbalance_data: bool, optional (default=False)
        Whether to handle data imbalance by oversampling the minority classes in the training set.

    Returns:
    --------
    tuple
        A tuple containing two dictionaries:
        - X: A dictionary with keys 'train', 'val', and 'test', each containing NumPy arrays of features.
        - y: A dictionary with keys 'train', 'val', and 'test', each containing NumPy arrays of labels.
    """
        
    #if handle_imbalance_data:
        #in_features_np, in_label_np = data_imbalance(in_features_np, in_label_np)
        
        
    X, y = {}, {}
    X['train'], X['test'], y['train'], y['test'] = train_test_split(in_features_np, in_label_np, stratify=in_label_np, train_size=train_size, random_state=random_state)
    X['test'], X['val'], y['test'], y['val'] = train_test_split(X['test'], y['test'], stratify=y['test'], train_size=0.5, random_state=random_state)
    
    if handle_imbalance_data:
        X['train'], y['train'] = data_imbalance(X['train'], y['train'])
    
    return X, y
    

def normalization(X, y, device):
    """
    Normalize input data.

    Parameters:
    -----------
    X: dict
        A dictionary containing NumPy arrays of input features for 'train', 'val', and 'test' splits.
    y: dict
        A dictionary containing NumPy arrays of input labels for 'train', 'val', and 'test' splits.
    device: torch.device
        The device (CPU or GPU) on which to store the normalized tensors.

    Returns:
    --------
    tuple
        A tuple containing three elements:
        - X_normalized: A dictionary with keys 'train', 'val', and 'test', each containing normalized input features as PyTorch tensors.
        - y: A dictionary with keys 'train', 'val', and 'test', each containing input labels as PyTorch tensors.
        - preprocess: The preprocessing transformation applied to the data (for possible future inverse transformation).
    """
        
    preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
    
    X = {
        k: torch.tensor(preprocess.transform(v), device=device)
        for k, v in X.items()}
        
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}
    
    return X, y, preprocess
