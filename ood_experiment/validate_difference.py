import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy.stats import ks_2samp, ttest_ind, shapiro

def validate_ood_data(
    X_train: np.array,
    X_ood: np.array,
    test: str = "welch",
    p_thresh: float = 0.01,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Validate OOD data by comparing it to the training (in-domain) data. For data to be OOD be assume a covariate shift
    to have taken place, i.e.

    1. p(x) =/= q(x)
    2. p(y|x) = q(y|x)

    We validate 1. by employing a Kolmogorov-Smirnov test along the feature dimension, checking whether the values are
    indeed coming from statistically different distributions.

    Parameters
    ----------
    X_train: np.array
        Training samples.
    X_ood: np.array
        OOD samples.
    test: str
        Significance test to use. Can be 'kolmogorov-smirnov' or 'welch'.
    p_thresh: float
        p-value threshold for KS test.
    feature_names: List[str]
        List of feature names.
    verbose: bool
        Print results to screen.

    Returns
    -------
    ks_p_values: np.array
        List of p-values from the Kolmogorov-Smirnov test for every feature.
    """
    assert test in (
        "kolmogorov-smirnov",
        "welch",
    ), "Invalid significance test specified."

    def _fraction_sig(p_values: np.array) -> Tuple[np.array, float]:
        p_values_sig = (p_values <= p_thresh).astype(int)
        fraction_sig = p_values_sig.mean()

        return p_values_sig, fraction_sig

    # Perform significance test for every feature dimension
    ks_p_values = []
    shapiro_train_p_values, shapiro_ood_p_values = [], []

    for d in range(X_train.shape[1]):
        X_train_d = X_train[~np.isnan(X_train[:, d]), d]
        X_ood_d = X_ood[~np.isnan(X_ood[:, d]), d]

        shapiro_train_p_values.append(
            shapiro(X_train_d)[1] if X_train_d.shape[0] > 2 else 1
        )
        shapiro_ood_p_values.append(shapiro(X_ood_d)[1] if X_ood_d.shape[0] > 2 else 1)

        if 0 in (X_train_d.shape[0], X_ood_d.shape[0]):
            p_value = 1
        else:
            test_func = (
                ks_2samp
                if test == "kolmogorov-smirnov"
                else lambda X, Y: ttest_ind(X, Y, equal_var=False)
            )
            _, p_value = test_func(X_train_d, X_ood_d)

        ks_p_values.append(p_value)

    ks_p_values = np.array(ks_p_values)

    ks_p_values_sig, percentage_sig = _fraction_sig(ks_p_values)
    _, sh_train_frac = _fraction_sig(np.array(shapiro_train_p_values))
    _, sh_ood_frac = _fraction_sig(np.array(shapiro_ood_p_values))

    if verbose:
        print(f"{sh_train_frac * 100:.2f} % of train feature are normally distributed.")
        print(f"{sh_ood_frac * 100:.2f} % of OOD feature are normally distributed.")
        print(
            f"{percentage_sig * 100:.2f} % of features ({ks_p_values_sig.sum()}) were stat. sig. different."
        )

    if feature_names is not None and percentage_sig > 0 and verbose:
        sorted_ks_p_values = list(
            sorted(zip(feature_names, ks_p_values), key=lambda t: t[1])
        )

        print("Most different features:")

        for i, (feat_name, p_val) in enumerate(sorted_ks_p_values):
            if p_val > p_thresh or i > 4:
                break

            print(f"{i+1}. {feat_name:<50} (p={p_val})")

    return ks_p_values, percentage_sig