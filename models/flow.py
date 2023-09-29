import numpy as np
import torch
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from torch import optim
from numpy.lib.function_base import average
from models.predictive_models import apply_model

class Flow():
    """
    Implements a Masked Autoregressive Flow (MAF).
    """

    def __init__(self,
                 device,
                 input_size: int,
                 hidden_features: int = 128,
                 num_layers: int = 5,
                 num_blocks_per_layer=2,
                 batch_norm_between_layers: bool = True,
                 lr: float = 1e-3,
                 **kwargs
                 ):
        """
        device: str
            The device on which to run the model (e.g., 'cpu' or 'cuda').
        input_size: int
            Dimensionality of the input.
        hidden_features: int
            Number of features in the hidden layers of transformations.
        num_layers: int
            Number of layers of transformations to be used.
        num_blocks_per_layer: int
            Number of blocks to be used in each transformation layer.
        batch_norm_between_layers: bool
            Specifies whether to use batch normalization between hidden layers.
        lr: float, optional
            The learning rate for model optimization. Default is 1e-3.

        **kwargs:
        use_residual_blocks: bool
            Specifies whether to use residual blocks containing masked linear modules. Note that residual blocks can't
             be used with random masks. Default value is True.
        use_random_masks=False,
            Specifies whether to use a random mask inside a linear module with masked weigth matrix. Note that residual
             blocks can't be used with random masks. Default value is False.
        use_random_permutations: bool
            Specifies whether features are shuffled at random after each transformation.
            Default value is False.
        activation
            Activation function in hidden layers. Default value torch.nn.functional.relu
        dropout_probability: float
            Dropout rate in hidden layers. Default value is 0.0
        batch_norm_within_layers: bool
            Specifies whether to use batch normalization within hidden layers. Default value is False.
        """

        self.model = MaskedAutoregressiveFlow(features=input_size,
                                              hidden_features=hidden_features,
                                              num_layers=num_layers,
                                              num_blocks_per_layer=num_blocks_per_layer,
                                              batch_norm_between_layers=batch_norm_between_layers,
                                              **kwargs).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def train(self, X_train, batch_size, n_epochs):
        """
        Train the novelty estimator.

        Parameters
        ----------
        X_train
            Training data.
        **kwargs:
            batch_size: int
                The batch size, default 128
            n_epochs: int
                The number of training epochs, default 30
        """

        ds_train = torch.utils.data.TensorDataset(X_train)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

        self.model.train()
        for epoch in range(n_epochs):
            average_loss, i = 0, 0
            for batch in dl_train:
                x = batch[0].to(self.device)
                loss = -self.model.log_prob(inputs=x).mean()
                average_loss += loss.item()
                i+=1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('Epoch:', epoch, 'Loss:', average_loss/i)
      
    def get_novelty_score(self, model, X):
        """
        Apply the novelty estimator to obtain a novelty score for the data.
        Returns scores that indicate negative log probability for each sample under the learned distribution.

        Parameters
        ----------
        X
            Samples to be scored.
        model
            A classification model to apply to the data. If None, no classification is performed.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing predicted labels and novelty scores as NumPy arrays.

        """

        # TODO: log_prob does not work for only one sample
        single_sample = False
        if X.shape[0] == 1:
            single_sample = True
            X = np.stack([X, X]).reshape(2, -1)

        with torch.no_grad():
            log_prob = self.model.log_prob(X).cpu().numpy()

        log_prob[np.isnan(log_prob)] = -np.inf
        log_prob[np.isinf(log_prob)] = log_prob[np.isinf(log_prob)].clip(min=-1e6, max=1e6)

        #if any(np.isnan(log_prob)):
        #    print("\t Warning: encountered NaN in the log probabilites (Flow)")

        #if any(np.isinf(log_prob)):
        #  print("\t Warning: encountered Inf in the log probabilites (Flow)")

        if single_sample:
            return log_prob[0]

        pred = np.zeros(log_prob.shape) #for compaatibility
        if model is not None:
            output = apply_model(model, X)
            pred = output.argmax(1).cpu().numpy()
        return pred, log_prob
