import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from models.predictive_models import apply_model

class LOF(LocalOutlierFactor):
    def __init__(self):
        LocalOutlierFactor.__init__(self, n_neighbors=19, algorithm="brute",  novelty=True)

    def train(self, X_train):
        """
        Train the novelty estimator.

        Parameters
        ----------
        X_train
            Training data.

        """
        super().fit(X_train.cpu().numpy())

    def get_novelty_score(self, model, X):
        """
        Apply the novelty estimator to obtain a novelty score for the data.
        Returns negative scores obtained from score_samples function. score_samples function assigns the large
        values to inliers and small values to outliers. Here, the negative  is used to get large values for outliers.

        Parameters
        ----------
        X: 
            Samples to be scored.
        model
            A classification model to apply to the data. If None, no classification is performed.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing predicted labels and novelty scores as NumPy arrays.

        """
        conf = self.score_samples(X.cpu().numpy())
        pred = np.zeros(conf.shape) #for compaatibility
        if model is not None:
            output = apply_model(model, X)
            pred = output.argmax(1).cpu().numpy()
            
        return pred, conf
