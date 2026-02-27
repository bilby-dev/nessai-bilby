"""Test the integration with bilby"""

import os
import signal

import bilby
import pytest

from nessai_bilby.model import BilbyModel


@pytest.fixture(params=[False, True])
def likelihood_constraint(request):
    return request.param


@pytest.fixture
def conversion_function():
    def _conversion_function(parameters, likelihood, prior):
        converted = parameters.copy()
        if "derived" not in converted:
            converted["derived"] = converted["m"] * converted["c"]
        return converted

    return _conversion_function


def run_sampler(
    likelihood,
    priors,
    outdir,
    conversion_function,
    sampler,
    n_pool=None,
    **kwargs,
):
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=sampler,
        outdir=str(outdir),
        save="hdf5",
        n_pool=n_pool,
        conversion_function=conversion_function,
        **kwargs,
    )
    return result


def test_sampling_nessai(
    bilby_likelihood,
    bilby_priors,
    conversion_function,
    tmp_path,
    likelihood_constraint,
    n_pool,
):
    outdir = tmp_path / "test_sampling_nessai"

    run_sampler(
        bilby_likelihood,
        bilby_priors,
        outdir,
        conversion_function,
        sampler="nessai",
        nlive=100,
        stopping=5.0,
        analytic_priors=True,
        seed=1234,
        nessai_likelihood_constraint=likelihood_constraint,
        n_pool=n_pool,
    )

    assert list(outdir.glob("*_nessai/*.png"))


def test_ensure_pool_is_used(
    bilby_likelihood, bilby_priors, tmp_path, caplog, monkeypatch
):
    # Patch the ModelClass to check if the pool is used
    from nessai_bilby import plugin

    class TestModel(BilbyModel):
        pool_used = False

        def batch_evaluate_log_likelihood(self, parameters):
            if self.pool is not None:
                TestModel.pool_used = True
            else:
                assert False, "Pool was not set in the model"
            return super().batch_evaluate_log_likelihood(parameters)

    monkeypatch.setattr(plugin, "BilbyModel", TestModel)

    with caplog.at_level("DEBUG", logger="nessai"):
        run_sampler(
            bilby_likelihood,
            bilby_priors,
            tmp_path / "test_pool_is_used",
            None,
            "nessai",
            n_pool=2,
            condition=10,
            nlive=100,
            nessai_likelihood_constraint=False,
        )
    assert "Using user specified pool" in caplog.text
    assert TestModel.pool_used, "Pool was not used during sampling"


def test_sampling_inessai(
    bilby_likelihood,
    bilby_priors,
    conversion_function,
    tmp_path,
    likelihood_constraint,
    n_pool,
):
    outdir = tmp_path / "test_sampling_inessai"

    run_sampler(
        bilby_likelihood,
        bilby_priors,
        outdir,
        conversion_function,
        sampler="inessai",
        n_pool=n_pool,
        nlive=100,
        min_samples=10,
        seed=1234,
        nessai_likelihood_constraint=likelihood_constraint,
    )


def test_sampling_nessai_plot(
    bilby_likelihood,
    bilby_priors,
    conversion_function,
    tmp_path,
):
    outdir = tmp_path / "test_sampling_nessai_plot"
    run_sampler(
        bilby_likelihood,
        bilby_priors,
        outdir,
        conversion_function,
        sampler="nessai",
        nlive=100,
        stopping=5.0,
        analytic_priors=True,
        seed=1234,
        nessai_likelihood_constraint=False,
        nessai_plot=False,
    )

    # Assert no png files in the output directory
    assert not list(outdir.glob("*_nessai/*.png"))


def test_interrupt_sampler(
    bilby_likelihood,
    bilby_priors,
    conversion_function,
    n_pool,
    tmp_path,
):
    outdir = tmp_path / "test_interrupt_sampler"
    interrupted = False

    def interrupt_once(_sampler):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            os.kill(os.getpid(), signal.SIGINT)

    label = "test_interrupt"

    with pytest.raises((SystemExit, KeyboardInterrupt)) as exc:
        run_sampler(
            bilby_likelihood,
            bilby_priors,
            outdir,
            conversion_function,
            "nessai",
            exit_code=5,
            resume=True,
            label=label,
            n_pool=n_pool,
            nlive=100,
            checkpoint_on_iteration=True,
            checkpoint_interval=1,
            checkpoint_callback=interrupt_once,
        )

    if isinstance(exc.value, SystemExit):
        assert exc.value.code == 5

    assert outdir.glob(f"{label}_nessai/*.pkl")
