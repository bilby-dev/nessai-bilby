"""Test the integration with bilby"""

import bilby
import pytest


@pytest.fixture(params=[False, True])
def likelihood_constraint(request):
    return request.param


def test_sampling_nessai(
    bilby_gaussian_likelihood_and_priors,
    tmp_path,
    likelihood_constraint,
    n_pool,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    outdir = tmp_path / "test_sampling_nessai"

    bilby.run_sampler(
        outdir=outdir,
        resume=False,
        plot=False,
        likelihood=likelihood,
        priors=priors,
        nlive=100,
        stopping=5.0,
        sampler="nessai",
        injection_parameters={"x": 0.0, "y": 0.0},
        analytic_priors=True,
        seed=1234,
        nessai_likelihood_constraint=likelihood_constraint,
        n_pool=n_pool,
    )
    # Assert plots are made
    assert list(outdir.glob("*_nessai/*.png"))


def test_sampling_inessai(
    bilby_gaussian_likelihood_and_priors,
    tmp_path,
    likelihood_constraint,
    n_pool,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    outdir = tmp_path / "test_sampling_inessai"

    bilby.run_sampler(
        outdir=outdir,
        resume=False,
        plot=False,
        likelihood=likelihood,
        priors=priors,
        nlive=100,
        min_samples=10,
        sampler="inessai",
        injection_parameters={"x": 0.0, "y": 0.0},
        seed=1234,
        nessai_likelihood_constraint=likelihood_constraint,
        n_pool=n_pool,
    )


def test_sampling_nessai_plot(
    bilby_gaussian_likelihood_and_priors,
    tmp_path,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    outdir = tmp_path / "test_sampling_nessai_plot"

    bilby.run_sampler(
        outdir=outdir,
        resume=False,
        plot=True,
        likelihood=likelihood,
        priors=priors,
        nlive=100,
        stopping=5.0,
        sampler="nessai",
        injection_parameters={"x": 0.0, "y": 0.0},
        analytic_priors=True,
        seed=1234,
        nessai_likelihood_constraint=False,
        nessai_plot=False,
        n_pool=None,
    )
    # Assert no png files in the output directory
    assert not list(outdir.glob("*_nessai/*.png"))
