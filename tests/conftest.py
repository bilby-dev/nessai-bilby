import bilby
import numpy as np
import pytest
from nessai.livepoint import reset_extra_live_points_parameters


def model(x, m, c):
    return m * x + c


def conversion_func(parameters):
    # d = |m| + |c|
    parameters["d"] = abs(parameters["m"]) + abs(parameters["c"])
    return parameters


@pytest.fixture()
def bilby_likelihood(rng):
    x = np.linspace(0, 10, 100)
    injection_parameters = dict(m=0.5, c=0.2)
    sigma = 1.0
    y = model(x, **injection_parameters) + rng.normal(0.0, sigma, len(x))
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma)
    return likelihood


@pytest.fixture()
def bilby_priors():
    priors = bilby.core.prior.PriorDict(conversion_function=conversion_func)
    priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
    priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
    priors["d"] = bilby.core.prior.Constraint(name="d", minimum=0, maximum=5)
    return priors


@pytest.fixture(autouse=True)
def reset_live_point_parameters():
    # Avoid issues when running standard and ins samplers in the same script.
    reset_extra_live_points_parameters()


@pytest.fixture()
def rng():
    return np.random.default_rng()


@pytest.fixture(params=[None, 2])
def n_pool(request):
    return request.param
