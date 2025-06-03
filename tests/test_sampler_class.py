import copy
import os
from unittest.mock import mock_open, patch

import bilby
import pytest

from nessai_bilby.plugin import ImportanceNessai, Nessai


@pytest.fixture(params=[Nessai, ImportanceNessai])
def SamplerClass(request):
    return request.param


@pytest.fixture()
def create_sampler(
    SamplerClass, bilby_gaussian_likelihood_and_priors, tmp_path
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    def create_fn(**kwargs):
        return SamplerClass(
            likelihood,
            priors,
            outdir=tmp_path / "outdir",
            label="test",
            use_ratio=False,
            **kwargs,
        )

    return create_fn


@pytest.fixture
def sampler(create_sampler):
    return create_sampler()


@pytest.fixture
def default_kwargs(sampler):
    expected = copy.deepcopy(sampler.default_kwargs)
    expected["output"] = os.path.join(
        sampler.outdir, f"{sampler.label}_{sampler.sampler_name}", ""
    )
    expected["seed"] = 12345
    return expected


@pytest.mark.parametrize(
    "key",
    bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs,
)
def test_translate_kwargs_nlive(create_sampler, key):
    sampler = create_sampler(**{key: 1000})
    assert sampler.kwargs["nlive"] == 1000


@pytest.mark.parametrize(
    "key",
    bilby.core.sampler.base_sampler.NestedSampler.npool_equiv_kwargs,
)
def test_translate_kwargs_npool(create_sampler, key):
    sampler = create_sampler(**{key: 2})
    assert sampler.kwargs["n_pool"] == 2


def test_split_kwargs(sampler):
    kwargs, run_kwargs = sampler.split_kwargs()
    assert "save" not in run_kwargs
    assert "plot" in run_kwargs


def test_translate_kwargs_no_npool(create_sampler):
    sampler = create_sampler()
    assert sampler.kwargs["n_pool"] == 1


def test_translate_kwargs_seed(create_sampler):
    sampler = create_sampler(sampling_seed=150914)
    assert sampler.kwargs["seed"] == 150914


@patch("builtins.open", mock_open(read_data='{"nlive": 4000}'))
def test_update_from_config_file(create_sampler):
    sampler = create_sampler(config_file="config_file.json")
    assert sampler.kwargs["nlive"] == 4000


def test_expected_outputs(SamplerClass):
    expected = os.path.join("outdir", f"test_{SamplerClass.sampler_name}", "")
    filenames, dirs = SamplerClass.get_expected_outputs(
        outdir="outdir",
        label="test",
    )
    assert len(filenames) == 0
    assert len(dirs) == 3
    assert dirs[0] == expected
    assert dirs[1] == os.path.join(expected, "proposal", "")
    assert dirs[2] == os.path.join(expected, "diagnostics", "")
