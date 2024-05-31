import bilby
from nessai.flowsampler import FlowSampler
from nessai_bilby.model import BilbyModel, BilbyModelLikelihoodConstraint
import numpy as np
import pytest


@pytest.fixture(params=[BilbyModel, BilbyModelLikelihoodConstraint])
def ModelClass(request):
    return request.param


def test_create_model(ModelClass, bilby_gaussian_likelihood_and_priors):
    likelihood, priors = bilby_gaussian_likelihood_and_priors
    priors = bilby.core.prior.PriorDict(priors)
    model = ModelClass(priors=priors, likelihood=likelihood)
    model.validate_bilby_likelihood()
    model.verify_model()


def test_sample_model_with_nessai(
    bilby_gaussian_likelihood_and_priors, tmp_path, ModelClass
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors
    priors = bilby.core.prior.PriorDict(priors)
    model = ModelClass(priors=priors, likelihood=likelihood)

    fs = FlowSampler(
        model=model,
        output=tmp_path,
        nlive=100,
        stopping=10,
    )
    fs.run()

    samples = fs.posterior_samples
    assert not np.isnan([samples[n] for n in samples.dtype.names]).any()
