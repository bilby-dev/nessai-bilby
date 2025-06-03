"""Interface for nessai in bilby"""

import os
import sys

from bilby.core.sampler.base_sampler import NestedSampler, signal_wrapper
from bilby.core.utils import (
    check_directory_exists_and_if_not_mkdir,
    load_json,
    logger,
)
from nessai.flowsampler import FlowSampler
from nessai.livepoint import live_points_to_array
from nessai.posterior import compute_weights
from nessai.utils.settings import get_all_kwargs, get_run_kwargs_list

try:
    from nessai.utils.logging import configure_logger
except ImportError:
    from nessai.utils.logging import setup_logger as configure_logger
import numpy as np
from pandas import DataFrame
from scipy.special import logsumexp

from .model import BilbyModel, BilbyModelLikelihoodConstraint


class Nessai(NestedSampler):
    """Bilby wrapper for the standard nested sampler in nessai.

    Nessai: (https://github.com/mj-will/nessai)

    All positional and keyword arguments passed to `run_sampler` are propagated
    to `nessai.flowsampler.FlowSampler`

    See the documentation for an explanation of the different kwargs.

    Documentation: https://nessai.readthedocs.io/
    """

    _default_kwargs = None
    _run_kwargs_list = None
    _importance_nested_sampler = False
    sampler_name = "nessai"
    sampling_seed_key = "seed"

    @property
    def run_kwargs_list(self):
        """List of kwargs used in the run method of :code:`FlowSampler`"""
        if not self._run_kwargs_list:
            self._run_kwargs_list = get_run_kwargs_list(
                importance_nested_sampler=self._importance_nested_sampler,
            )
            ignored_kwargs = ["save"]
            for ik in ignored_kwargs:
                if ik in self._run_kwargs_list:
                    self._run_kwargs_list.remove(ik)
        return self._run_kwargs_list

    @property
    def default_kwargs(self):
        """Default kwargs for nessai.

        Retrieves default values from nessai directly and then includes any
        bilby specific defaults. This avoids the need to update bilby when the
        defaults change or new kwargs are added to nessai.

        Includes the following kwargs that are specific to bilby:

        - :code:`nessai_log_level`: allows setting the logging level in nessai
        - :code:`nessai_logging_stream`: allows setting the logging stream
        - :code:`nessai_plot`: allows toggling the plotting in FlowSampler.run
        - :code:`nessai_likelihood_constraint`: allows toggling between
          including the prior constraints in likelihood or the prior.
        """
        if not self._default_kwargs:
            kwargs = get_all_kwargs(
                importance_nested_sampler=self._importance_nested_sampler,
            )

            # Defaults for bilby that will override nessai defaults
            bilby_defaults = dict(
                output=None,
                exit_code=self.exit_code,
                nessai_log_level=None,
                nessai_logging_stream="stdout",
                nessai_plot=True,
                nessai_likelihood_constraint=True,
                plot_posterior=False,  # bilby already produces a posterior plot
                log_on_iteration=False,  # Use periodic logging by default
                logging_interval=60,  # Log every 60 seconds
            )
            kwargs.update(bilby_defaults)
            # Kwargs that cannot be set in bilby
            remove = [
                "save",
                "signal_handling",
                "importance_nested_sampler",
            ]
            for k in remove:
                if k in kwargs:
                    kwargs.pop(k)
            self._default_kwargs = kwargs
        return self._default_kwargs

    def log_prior(self, theta):
        """

        Parameters
        ----------
        theta: list
            List of sampled values on a unit interval

        Returns
        -------
        float: Joint ln prior probability of theta

        """
        return self.priors.ln_prob(theta, axis=0)

    def split_kwargs(self):
        """Split kwargs into configuration and run time kwargs"""
        kwargs = self.kwargs.copy()
        run_kwargs = {}
        for k in self.run_kwargs_list:
            run_kwargs[k] = kwargs.pop(k)
        run_kwargs["plot"] = kwargs.pop("nessai_plot")
        return kwargs, run_kwargs

    def get_posterior_weights(self):
        """Get the posterior weights for the nested samples"""

        _, log_weights = compute_weights(
            np.array(self.fs.nested_samples["logL"]),
            np.array(self.fs.ns.state.nlive),
        )
        w = np.exp(log_weights - logsumexp(log_weights))
        return w

    def get_nested_samples(self):
        """Get the nested samples dataframe"""
        ns = DataFrame(self.fs.nested_samples)
        ns.rename(
            columns=dict(
                logL="log_likelihood", logP="log_prior", it="iteration"
            ),
            inplace=True,
        )
        return ns

    def update_result(self):
        """Update the result object."""

        # Manually set likelihood evaluations because parallelisation breaks the counter
        self.result.num_likelihood_evaluations = (
            self.fs.ns.total_likelihood_evaluations
        )

        self.result.sampling_time = self.fs.ns.sampling_time
        self.result.samples = live_points_to_array(
            self.fs.posterior_samples, self.search_parameter_keys
        )
        self.result.log_likelihood_evaluations = self.fs.posterior_samples[
            "logL"
        ]
        self.result.nested_samples = self.get_nested_samples()
        self.result.nested_samples["weights"] = self.get_posterior_weights()
        self.result.log_evidence = self.fs.log_evidence
        self.result.log_evidence_err = self.fs.log_evidence_error

    @signal_wrapper
    def run_sampler(self):
        """Run the sampler.

        Nessai is designed to be ran in two stages, initialise the sampler
        and then call the run method with additional configuration. This means
        there are effectively two sets of keyword arguments: one for
        initializing the sampler and the other for the run function.
        """
        kwargs, run_kwargs = self.split_kwargs()

        # Setup the logger for nessai, use nessai_log_level if specified, else
        # use the level of the bilby logger.
        nessai_log_level = kwargs.pop("nessai_log_level")
        if nessai_log_level is None or nessai_log_level == "bilby":
            nessai_log_level = logger.getEffectiveLevel()
        nessai_logging_stream = kwargs.pop("nessai_logging_stream")

        configure_logger(
            self.outdir,
            label=self.label,
            log_level=nessai_log_level,
            stream=nessai_logging_stream,
        )

        likelihood_constraint = kwargs.pop("nessai_likelihood_constraint")

        if likelihood_constraint:
            ModelClass = BilbyModelLikelihoodConstraint
        else:
            ModelClass = BilbyModel

        model = ModelClass(
            priors=self.priors,
            likelihood=self.likelihood,
            use_ratio=self.use_ratio,
        )

        # Configure the sampler
        self.fs = FlowSampler(
            model,
            signal_handling=False,  # Disable signal handling so it can be handled by bilby
            importance_nested_sampler=self._importance_nested_sampler,
            **kwargs,
        )
        # Run the sampler
        self.fs.run(**run_kwargs)

        # Update the result
        self.update_result()

        return self.result

    def _translate_kwargs(self, kwargs):
        """Translate the keyword arguments"""
        super()._translate_kwargs(kwargs)
        if "nlive" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["nlive"] = kwargs.pop(equiv)
        if "n_pool" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["n_pool"] = kwargs.pop(equiv)
            if "n_pool" not in kwargs:
                kwargs["n_pool"] = self._npool

    def _verify_kwargs_against_default_kwargs(self):
        """Verify the keyword arguments"""
        if "config_file" in self.kwargs:
            d = load_json(self.kwargs["config_file"], None)
            self.kwargs.update(d)
            self.kwargs.pop("config_file")

        if not self.kwargs["plot"]:
            self.kwargs["plot"] = self.plot

        if not self.kwargs["output"]:
            self.kwargs["output"] = os.path.join(
                self.outdir, f"{self.label}_{self.sampler_name}", ""
            )

        check_directory_exists_and_if_not_mkdir(self.kwargs["output"])
        NestedSampler._verify_kwargs_against_default_kwargs(self)

    def write_current_state(self):
        """Write the current state of the sampler"""
        self.fs.ns.checkpoint()

    def write_current_state_and_exit(self, signum=None, frame=None):
        """
        Overwrites the base class to make sure that :code:`Nessai` terminates
        properly.
        """
        if hasattr(self, "fs"):
            self.fs.terminate_run(code=signum)
        else:
            logger.warning("Sampler is not initialized")
        self._log_interruption(signum=signum)
        sys.exit(self.exit_code)

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Get lists of the expected outputs directories and files.

        These are used by :code:`bilby_pipe` when transferring files via HTCondor.

        Parameters
        ----------
        outdir : str
            The output directory.
        label : str
            The label for the run.

        Returns
        -------
        list
            List of file names. This will be empty for nessai.
        list
            List of directory names.
        """
        dirs = [os.path.join(outdir, f"{label}_{cls.sampler_name}", "")]
        dirs += [
            os.path.join(dirs[0], d, "") for d in ["proposal", "diagnostics"]
        ]
        filenames = []
        return filenames, dirs

    def _setup_pool(self):
        pass


class ImportanceNessai(Nessai):
    """Bilby wrapper for the importance nested sampler in nessai.

    Nessai: (https://github.com/mj-will/nessai)

    See the documentation for an explanation of the different kwargs.

    Documentation: https://nessai.readthedocs.io/
    """

    _importance_nested_sampler = True
    sampler_name = "inessai"

    @property
    def external_sampler_name(self):
        return "nessai"

    def get_posterior_weights(self):
        """Get the posterior weights for the nested samples"""
        log_w = self.fs.nested_samples["logL"] + self.fs.nested_samples["logW"]
        w = np.exp(log_w - logsumexp(log_w))
        return w
