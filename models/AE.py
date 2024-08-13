import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from typing import List, Optional
from models.predictive_models import apply_model

class Encoder(nn.Module):
    """The encoder module, which encodes an input into the latent space.

    Parameters
    ----------
    input_dim: int
        The input dimensionality.
    hidden_dims: List[int]
        A list with the sizes of the hidden layers.
    latent_dim: int
        The size of the latent space.
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim, device = 'cpu'):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [100, 50]
        
        if latent_dim is None:
            latent_dim = 20
            
        self.layers = []
        architecture = [input_dim] + hidden_dims + [latent_dim]

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*self.layers).to(device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of encoder. Returns latent representation.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.
        """
        return self.encoder(input_tensor)


class Decoder(nn.Module):
    """The decoder module, which decodes the latent representation back to the space of
    the input data.

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input
    hidden_dims: List[int]
        A list with the sizes of the hidden layers.
    latent_dim: int
        The size of the latent space.
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, device):
        super().__init__()
        architecture = [latent_dim] + hidden_dims + [input_dim]
        self.layers = []

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*self.layers).to(device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of decoder. Returns reconstructed input data.

        Parameters
        ----------
        input_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.
        """
        return self.decoder(input_tensor)


class AEModule(nn.Module):
    """The Pytorch module of a  Autoencoder, consisting of an equally-sized encoder and
    decoder.

    Parameters
    ----------
    input_size: int
        The dimensionality of the input, assumed to be a 1-d vector.
    hidden_dims: List[int]
        A list of integers, representing the hidden dimensions of the encoder and decoder. These
        hidden dimensions are the same for the encoder and the decoder.
    latent_dim: int
        The dimensionality of the latent space.
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, input_size: int, hidden_dims: List[int], latent_dim: int, device):
        super().__init__()

        self.z_dim = latent_dim
        self.encoder = Encoder(input_size, hidden_dims, latent_dim, device)
        self.decoder = Decoder(input_size, hidden_dims, latent_dim, device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform an encoding and decoding step and return the
        reconstruction error for the given batch.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the VAE.

        Returns
        -------
        reconstr_error: torch.Tensor
            The reconstruction error.
        """

        input_tensor = input_tensor.float()
        # encoding
        z = self.encoder(input_tensor)

        # decoding
        reconstruction = self.decoder(z)

        # calculating losses
        mse = torch.nn.MSELoss(reduction="none")
        reconstr_error = mse(reconstruction, input_tensor).mean(dim=1)

        return reconstr_error
    
    def train(self):
        self.encoder.encoder.train()
        self.decoder.decoder.train()
    
    def eval(self):
        self.encoder.encoder.eval()
        self.decoder.decoder.eval()
  

class AE():
    def __init__(self, hidden_sizes: List[int], input_size: int,
                 latent_dim: int, device, lr: float = 1e-3):

        """The autoencoder class that handles training and reconstruction.
    
        Parameters
        ----------
        hidden_sizes: List[int]
            A list with the sizes of the hidden layers.
        input_size: int
            The dimensionality of the input
        latent_dim: int
            The size of the latent space.
        device: str
            The device on which to run the model (e.g., 'cpu' or 'cuda').
        lr: float, optional
            The learning rate for model optimization. Default is 1e-3.
        """
                     
        if hidden_sizes is None:
            hidden_sizes = [100, 50]
        
        if latent_dim is None:
            latent_dim = 20
      
        self.model = AEModule(input_size=input_size, hidden_dims=hidden_sizes, latent_dim=latent_dim, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
    
    def train(self, X_train, batch_size=32, n_epochs=5):
        """
        Train the autoencoder model on the given data.

        Parameters
        ----------
        X_train: torch.Tensor
            The training data as a torch.Tensor.
        batch_size: int, optional
            The batch size for training. Default is 32.
        n_epochs: int, optional
            The number of training epochs. Default is 5.
        """

        print('Start training ...')
        
        train_dataset = X_train.float()
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        #val_dataset = X_val.float()
        #self.val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


        for epoch in range(n_epochs):
            self.model.train()
            average_reconstruction_error, i = 0, 0
            for i, batch in enumerate(train_data):
                batch = batch.to(self.device)
                reconstruction_error = self.model(batch).mean()
                average_reconstruction_error += reconstruction_error.item()

                self.optimizer.zero_grad()
                reconstruction_error.backward()
                self.optimizer.step()

            average_epoch_rec_error = average_reconstruction_error / (i + 1)
            print('Epoch:', epoch, 'Average reconstruction error:', average_epoch_rec_error)

    @torch.no_grad()
    def postprocess(self, model, data):
        """
        Postprocess data using the trained autoencoder model to measure the novelty score.

        Parameters
        ----------
        model: torch.nn.Module or None
            A classification model to apply to the data. If None, no classification is performed.
        data: torch.Tensor
            The data is to be postprocessed as a torch.Tensor.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing predicted labels and novelty scores as NumPy arrays.
        """
        
        conf = -self.model(data)
        pred = np.zeros(conf.shape) #for compaatibility
        
        if model is not None:
            output = apply_model(model, data)
            pred = output.argmax(1).cpu().numpy()

        return pred, conf.cpu().numpy()
