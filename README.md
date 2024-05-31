# nessai-bilby

Interface and plugin for using `nessai` in `bilby`.

This plugin provides two samplers that can be used in `bilby`:

- `nessai`: the standard nested sampler from `nessai`
- `inessai`: the importance nested sampler from nessai


It also provides a means to use `bilby` likelihoods and priors directly in
`nessai`, see [using bilby likelihoods in nessai](#using-bilby-likelihoods-in-nessai)

## Installation

The package can be installed using pip

```
pip install nessai-bilby
```
or conda

```
conda install conda-forge::nessai-bilby
```

However, we recommend following installing PyTorch manually to ensure the
correct device support.

**Note:** this plugin requires "bilby>=2.3.0".


## Usage

### In bilby

One `nessai-bilby` is installed, both samplers can be used directly in `bilby`
via the `run_sampler` function. See the bilby documentation for more details
on how to run different samplers.


### Using bilby likelihoods in nessai

`nessai-bilby` also provides two model classes that allow bilby likelihood and
priors to be used directly with nessai:

- `nessai_bilby.model.BilbyModel`: 
- `nessai_bilby.model.BilbyModelLikelihoodConstraint`:


Either model can be used by creating an instance of the model and running `nessai` as usual:

```python
from nessai.flowsampler import FlowSampler
from nessai_bilby.model import BilbyModel

likelihood = ...    # bilby likelihood object
priors = ... # bilby PriorDict

model = BilbyModel(
    priors=priors,
    likelihood=likelihood,
    use_ratio=True    # Whether to use the log-likelihood ratio
)

fs = FlowSampler(
    model,
    ...,
)

fs.run()
```

## Citing

If you use `nessai-bilby`, please cite the `nessai` and `bilby` code bases and the corresponding papers.
