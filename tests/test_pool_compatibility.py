"""Bilby initializer compatibility and actual multiprocessing regression."""

import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from nessai_bilby.plugin import (
    ImportanceNessai,
    Nessai,
    _initialize_global_variables,
)


@pytest.mark.parametrize("modern", [True, False])
def test_initializer_locations(monkeypatch, modern):
    import bilby.core.sampler.base_sampler as base
    import nessai.utils.multiprocessing as multiprocessing

    initializer = Mock()
    parallel = ModuleType("bilby.core.utils.parallel")
    if modern:
        parallel.initialize_global_variables = initializer
    monkeypatch.setitem(sys.modules, "bilby.core.utils.parallel", parallel)
    legacy = Mock() if modern else initializer
    monkeypatch.setattr(
        base, "_initialize_global_variables", legacy, raising=False
    )
    nessai_initializer = Mock()
    monkeypatch.setattr(
        multiprocessing, "initialise_pool_variables", nessai_initializer
    )
    _initialize_global_variables(
        "likelihood", "priors", ["x"], True, {}, "model"
    )
    initializer.assert_called_once_with(
        likelihood="likelihood",
        priors="priors",
        search_parameter_keys=["x"],
        use_ratio=True,
        parameters={},
    )
    nessai_initializer.assert_called_once_with("model")
    if modern:
        legacy.assert_not_called()


@pytest.mark.parametrize("sampler_class", [Nessai, ImportanceNessai])
def test_real_pool_likelihood(
    sampler_class, bilby_likelihood, bilby_priors, tmp_path
):
    from nessai.utils.multiprocessing import log_likelihood_wrapper

    from nessai_bilby.model import BilbyModelLikelihoodConstraint

    sampler = sampler_class(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        outdir=str(tmp_path),
        label="pool-test",
        n_pool=2,
        use_ratio=False,
    )
    model = BilbyModelLikelihoodConstraint(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
    )
    point = model.new_point()
    expected = model.log_likelihood(point)
    try:
        sampler._setup_pool(model)
        actual = sampler.pool.apply_async(
            log_likelihood_wrapper, (point,)
        ).get(20)
        np.testing.assert_allclose(actual, expected)
    finally:
        if getattr(sampler, "pool", None) is not None:
            sampler.pool.terminate()
            sampler.pool.join()
