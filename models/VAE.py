from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.nn import functional as F
import torch.distributions as dist
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from models.predictive_models import apply_model

class Encoder(nn.Module):
    """The encoder module, which encodes an input into the latent space.

    Parameters
    ----------
    hidden_sizes: List[int]
        A list with the sizes of the hidden layers.
    input_size: int
        The input dimensionality.
    latent_dim: int
        The size of the latent space.
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, hidden_sizes: List[int], input_size: int, latent_dim: int, device):
        super().__init__()
        architecture = [input_size] + hidden_sizes
        self.layers = []

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())

        self.hidden = nn.Sequential(*self.layers).to(device)
        self.mean = nn.Linear(architecture[-1], latent_dim).to(device)
        self.log_var = nn.Linear(architecture[-1], latent_dim).to(device)
        self.device = device

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass of encoder. Returns mean and standard deviation corresponding to
        an independent Normal distribution.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.
        """
        h = self.hidden(input_tensor)
        mean = self.mean(h)
        log_var = self.log_var(h)
        std = torch.sqrt(torch.exp(log_var))

        return mean, std

    def train(self):
        self.hidden.train()
        self.mean.train()
        self.log_var.train()

    def eval(self):
        self.hidden.eval()
        self.mean.eval()
        self.log_var.eval()


class Decoder(nn.Module):
    """
    The decoder module, which decodes a sample from the latent space back to the space of
    the input data.

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
    """

    def __init__(self, hidden_sizes: List[int], input_size: int, latent_dim: int, device):
        super().__init__()
        architecture = [latent_dim] + hidden_sizes
        self.layers = []

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())

        self.hidden = nn.Sequential(*self.layers).to(device)
        self.mean = nn.Linear(architecture[-1], input_size).to(device)
        #self.log_var = nn.Linear(architecture[-1], input_size).to(device)
        self.device = device

    def forward(
            self, latent_tensor: torch.Tensor, reconstruction_mode: str = "mean") -> torch.Tensor:
        """Perform forward pass of decoder. Returns mean and standard deviation corresponding
        to an independent Normal distribution.

        Parameters
        ----------
        latent_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.
        reconstruction_mode: str
            Specify the way that a sample should be reconstructed. 'mean' simply returns the mean of p(x|z), 'sample'
            samples from the same distribution.
        """
        assert reconstruction_mode in ["mean", "sample"], (
            "Invalid reconstruction mode given, must be 'mean' or "
            f"'sample', '{reconstruction_mode}' found."
        )

        h = self.hidden(latent_tensor)
        mean = self.mean(h)
        #log_var = self.log_var(h)
        #std = torch.sqrt(torch.exp(log_var))

        # Just return the mean
        if reconstruction_mode == "mean":
            return mean

        # Sample
        else:
            eps = torch.randn(mean.shape)
            return mean + eps * std

    def reconstruction_error(
            self, input_tensor: torch.Tensor, latent_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the log probability of the original data sample under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original data sample.
        latent_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the input under the decoder's distribution.
        """
        h = self.hidden(latent_tensor)
        mean = self.mean(h)

        reconstr_error = F.mse_loss(mean, input_tensor, reduction='none').mean(dim=1)

        #log_var = self.log_var(h)
        #std = torch.sqrt(torch.exp(log_var))
        #distribution = dist.independent.Independent(dist.normal.Normal(mean, std), 0)
        #reconstr_error = -distribution.log_prob(input_tensor).sum(dim=1)

        return reconstr_error

    def train(self):
        self.hidden.train()
        self.mean.train()
        #self.log_var.train()

    def eval(self):
        self.hidden.eval()
        self.mean.eval()
        #self.log_var.eval()


class VAEModule(nn.Module):
    """The Pytorch module of a Variational Autoencoder, consisting of an equally-sized encoder and
    decoder. This module works for continuous distributions. In case of discrete distributions,
    it has to be adjusted (outputting a Bernoulli distribution instead of independent Normal).

    Parameters
    ----------
    input_size: int
        The dimensionality of the input, assumed to be a 1-d vector.
    hidden_sizes: List[int]
        A list of integers, representing the hidden dimensions of the encoder and decoder. These
        hidden dimensions are the same for the encoder and the decoder.
    latent_dim: int
        The dimensionality of the latent space.
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, hidden_sizes: List[int], input_size: int, latent_dim: int, device):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(hidden_sizes, input_size, latent_dim, device)
        self.decoder = Decoder(hidden_sizes, input_size, latent_dim, device)
        self.device = device

    def forward(self, input_tensor: torch.Tensor, reconstr_error_weight: float, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform an encoding and decoding step and return the
        reconstruction error, KL-divergence and negative average elbo for the given batch.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the VAE.
        reconstr_error_weight: float
            A factor which is multiplied with the reconstruction error, to weigh this term in
            the overall loss function.
        beta: float
            Weighting term for the KL divergence.

        Returns
        -------
        reconstr_error: torch.Tensor
            The reconstruction error.
        kl: torch.Tensor
            The KL-divergence.
        average_negative_elbo: torch.Tensor
            The negative ELBO averaged over the batch.
        """

        input_tensor = input_tensor.float()
        # encoding
        mean, std = self.encoder(input_tensor)
        eps = torch.randn(mean.shape, device=self.device)
        z = mean + eps * std

        # decoding
        reconstr_error = self.decoder.reconstruction_error(input_tensor, z)
        d = mean.shape[1]

        # Calculating the KL divergence of the two independent Gaussians (closed-form solution)
        kl = 0.5 * torch.sum(
            std - torch.ones(d, device=self.device) - torch.log(std + 1e-8) + mean * mean, dim=1
        )
        average_negative_elbo = torch.mean(
            reconstr_error_weight * reconstr_error + kl * beta
        )

        return reconstr_error, kl, average_negative_elbo

    def get_reconstruction_error_grad(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Return the gradient of log p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input for which the gradient of the reconstruction error should be computed.

        Returns
        -------
        torch.Tensor
            Gradient of reconstruction error w.r.t. the input.
        """
        model_state = self.encoder.training
        self.encoder.train()
        self.decoder.train()

        input_tensor = input_tensor.float()
        input_tensor.requires_grad = True

        # Encoding
        h = self.encoder.hidden(input_tensor)
        mean = self.encoder.mean(h)

        # Decoding
        reconstr_error = self.decoder.reconstruction_error(input_tensor, mean)
        # Compute separate grad for each bach instance
        reconstr_error.backward(gradient=torch.ones(reconstr_error.shape))
        grad = input_tensor.grad

        # Reset model state to what is was before
        self.encoder.training = model_state
        self.decoder.training = model_state

        return grad

    def get_reconstruction_grad_magnitude(
            self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve the l2-norm of the gradient of log(x|z) w.r.t to the input.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input for which the magnitude of the gradient w.r.t. the reconstruction error should be computed.

        Returns
        -------
        torch.Tensor
            Magnitude of gradient of reconstruction error wr.t. the input.
        """
        norm = torch.norm(self.get_reconstruction_error_grad(input_tensor), dim=1)

        return norm

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

class VAE():
    """
    The VAE class that handles training and reconstruction.

    Parameters
    ----------
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    input_size: int
        The dimensionality of the input
    hidden_sizes: List[int]
        A list with the sizes of the hidden layers.
    latent_dim: int
        The size of the latent space.
    beta: float
        Weighting term for the KL-divergence.
    anneal: bool
        Option to indicate whether KL-divergence should be annealed.
    """

    def __init__(
            self,
            device,
            hidden_sizes: List[int],
            input_size: int,
            latent_dim: int,
            beta: float = 2.0,
            anneal: bool = True,
            lr: float = 1e-3,
            reconstr_error_weight: float = 0.15):
        
        if hidden_sizes is None:
            hidden_sizes = [100, 50, 50]
        
        if latent_dim is None:
            latent_dim = 10

        self.model = VAEModule(input_size=input_size, hidden_sizes=hidden_sizes, latent_dim=latent_dim, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.reconstr_error_weight = reconstr_error_weight
        self.beta = beta
        self.anneal = anneal
        self.device = device

    def train(self, X_train, batch_size=32, n_epochs=5):
        """
        Train the model on the given data.

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

        for epoch in range(n_epochs):
            self.model.train()
            self._epoch_iter(train_data, epoch, n_epochs)



    def get_novelty_score(self, model, X):
        """
        Apply the novelty estimator to obtain a novelty score for the data.

        Parameters
         ----------
        X: np.array

        Returns
         -------
        np.array
            Novelty scores for each sample.
        """

        scoring_function = "reconstr_err"

        if scoring_function == "reconstr_err":
            conf = self._get_reconstr_error(X)

        elif scoring_function == "latent_prob":
            conf = self._get_latent_prob(X)

        elif scoring_function == "latent_prior_prob":
            conf = self._get_latent_prior_prob(X)

        elif scoring_function == "reconstr_err_grad":
            conf = self._get_reconstruction_grad_magnitude(X)
        
        conf[conf.isnan()] = 1e3
        conf = conf.clamp(max=1e3)
        conf = -conf.detach().cpu().numpy()
        
        pred = np.zeros(conf.shape) #for compaatibility
        if model is not None:
            output = apply_model(model, X)
            pred = output.argmax(1).cpu().numpy()
            
        return pred, conf

    def _epoch_iter(self, data: torch.utils.data.DataLoader, current_epoch, n_epochs) -> float:
        """Iterate through the data once and return the average negative ELBO. If the train data
        is fed,the model parameters are updated. If the validation data is fed, only the average
        elbo is calculated and no parameter update is performed.

        Parameters
        ----------
        data: torch.utils.data.DataLoader
            The dataloader of the train or validation set.
        current_epoch: int
            Number of current epoch.
        n_epochs: int
            Total number of epochs.

        Returns
        -------
        average_epoch_elbo: float
            The negative ELBO averaged over the epoch.
        """

        average_epoch_elbo, i = 0, 0

        for i, batch in enumerate(tqdm(data)):

            if self.anneal:
                beta = self._get_beta(
                    target_beta=self.beta,
                    current_epoch=current_epoch,
                    current_iter=i,
                    n_epochs=n_epochs,
                    n_iters=len(data))
            else:
                beta = self.beta

            batch = batch.to(self.device)
            _, _, average_negative_elbo = self.model(batch, reconstr_error_weight=self.reconstr_error_weight, beta=beta)
            average_epoch_elbo += average_negative_elbo

            if torch.isnan(average_negative_elbo):
                raise ValueError("ELBO is nan.")
            average_negative_elbo.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 4)
            self.optimizer.step()
            self.optimizer.zero_grad()

        average_epoch_elbo = average_epoch_elbo / (i + 1)
        print('Epoch:', current_epoch, 'Loss:', average_epoch_elbo.item())

    @staticmethod
    def _get_beta(
            target_beta: float,
            current_epoch: int,
            current_iter: int,
            n_epochs: int,
            n_iters: int,
            saturation_percentage: float = 0.4,
    ) -> float:
        """
        Get the current beta term.

        Parameters
        ----------
        target_beta: float
            Target value for beta.
        current_epoch: int
            Current epoch number.
        current_iter: int
            Number of interations in current epoch.
        n_epochs: int
            Total number of epochs.
        n_iters:
            Number of iterations per epoch.
        saturation_percentage: float
            Percentage of total iterations after which the target_beta value should be reached.

        Returns
        -------
        float
            Annealed beta value.
        """
        total_iters = n_epochs * n_iters
        current_total_iter = current_epoch * n_iters + current_iter
        annealed_beta = (
                min(current_total_iter / (saturation_percentage * total_iters), 1)
                * target_beta
        )

        return annealed_beta

    def _get_reconstr_error(
            self, data, n_samples: int = 10
    ) -> np.ndarray:
        """Calculate the reconstruction error for some data (assumed to be a numpy array).
        The reconstruction error is averaged over a number of samples.

        Parameters
        ----------
        data
            The data of which we want to know the reconstruction error.
        n_samples: int, default 10
            The number of samples to take to calculate the average reconstruction error.

        Returns
        -------
        avg_reconstruction_error: np.ndarray
            The average reconstruction error for each item in the data.
        """
        self.model.eval()
        reconstructions = []

        for i in range(n_samples):
            reconstr_error, _, _ = self.model(data, reconstr_error_weight=self.reconstr_error_weight)
            reconstructions.append(reconstr_error.unsqueeze(0).detach())

        concatenated_rec = torch.cat(reconstructions, axis=0)
        avg_reconstruction_error = torch.mean(concatenated_rec, axis=0)

        return avg_reconstruction_error

    def _get_latent_encoding(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """Encode the data to the latent space. The latent representation is defined by a
        mean and standard deviation corresponding to an independent Normal distribution.

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent encodings.

        Returns
        -------
        z: np.ndarray
            The latent encoding of the data.
        """
        self.model.eval()
        mean, std = self.model.encoder(data.unsqueeze(0).float())
        mean = mean.squeeze(0).detach().numpy()
        std = std.squeeze(0).detach().numpy()

        eps = np.random.randn(*mean.shape, device=self.device)
        z = mean + eps * std

        return z

    def _get_latent_prior_prob(self, data) -> np.ndarray:
        """
        Get the probability of the latent representation corresponding to an input according
        to the latent space prior p(z).

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent probabilities.

        Returns
        -------
        np.ndarray
            Log probabilities of latent representations.
        """
        self.model.eval()
        mean, _ = self.model.encoder(data)

        # For VAE, the latent space is an isotropic gaussian
        distribution = dist.independent.Independent(dist.normal.Normal(0, 1), 0)
        latent_prob = distribution.log_prob(mean).sum(dim=1).detach().numpy()

        return latent_prob

    def _get_latent_prob(self, data) -> np.ndarray:
        """
        Get the probability of the latent representation corresponding to an input according
        to q(z|x).

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent probabilities.

        Returns
        -------
        np.ndarray
            Log probabilities of latent representations.
        """
        self.model.eval()
        mean, std = self.model.encoder(data)

        # For VAE, the latent space is an isotropic gaussian
        distribution = dist.independent.Independent(dist.normal.Normal(mean, std), 0)
        latent_prob = distribution.log_prob(mean).sum(dim=1).detach().numpy()

        return latent_prob

    def _get_reconstruction_grad_magnitude(self, data) -> np.ndarray:
        """
        Retrieve the l2-norm of the gradient of log(x|z) w.r.t to the input.

        Parameters
        ----------
        data: data: np.ndarray
            Input for which the magnitude of the gradient w.r.t. the reconstruction error should be computed.

        Returns
        -------
        data: np.ndarray
            Magnitude of gradient of reconstruction error wr.t. the input.
        """
        grad_magnitude = self.model.get_reconstruction_grad_magnitude(data)
        grad_magnitude = grad_magnitude.detach().numpy()

        return grad_magnitude
