import bilby
import numpy as np
import pytest


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

