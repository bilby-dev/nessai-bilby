import bilby
import numpy as np
import pytest
from nessai.livepoint import reset_extra_live_points_parameters


@pytest.fixture
def bilby_gaussian_likelihood_and_priors():
    class GaussianLikelihood(bilby.Likelihood):
        def __init__(self):
            """A very simple Gaussian likelihood"""
            super().__init__(parameters={"x": None, "y": None})

        def log_likelihood(self):
            """Log-likelihood."""
            return -0.5 * (
                self.parameters["x"] ** 2.0 + self.parameters["y"] ** 2.0
            ) - np.log(2.0 * np.pi)

    likelihood = GaussianLikelihood()
    priors = dict(
        x=bilby.core.prior.Uniform(-10, 10, "x"),
        y=bilby.core.prior.Uniform(-10, 10, "y"),
    )
    return likelihood, priors


@pytest.fixture(autouse=True)
def reset_live_point_parameters():
    # Avoid issues when running standard and ins samplers in the same script.
    reset_extra_live_points_parameters()


@pytest.fixture()
def rng():
    return np.random.default_rng()
