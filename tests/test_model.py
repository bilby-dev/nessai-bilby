import bilby
import numpy as np
import pytest
from nessai.flowsampler import FlowSampler

from nessai_bilby.model import BilbyModel, BilbyModelLikelihoodConstraint


@pytest.fixture(params=[BilbyModel, BilbyModelLikelihoodConstraint])
def ModelClass(request):
    return request.param


def test_create_model(ModelClass, bilby_likelihood, bilby_priors, rng):
    priors = bilby.core.prior.PriorDict(bilby_priors)
    model = ModelClass(priors=priors, likelihood=bilby_likelihood)
    # Do this rather than using `set_rng` to ensure backwards compatibility
    model.set_rng(rng)
    model.validate_bilby_likelihood()
    model.verify_model()


def test_sample_model_with_nessai(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    ModelClass,
):
    priors = bilby.core.prior.PriorDict(bilby_priors)
    model = ModelClass(priors=priors, likelihood=bilby_likelihood)

    fs = FlowSampler(
        model=model,
        output=tmp_path,
        nlive=100,
        stopping=10,
    )
    fs.run()

    samples = fs.posterior_samples
    assert not np.isnan([samples[n] for n in samples.dtype.names]).any()
