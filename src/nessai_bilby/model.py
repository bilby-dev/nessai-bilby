"""Utilities for using nessai with external packages"""

from typing import TYPE_CHECKING

import numpy as np
from nessai.livepoint import dict_to_live_points
from nessai.model import Model

if TYPE_CHECKING:
    from bilby.core.likelihood import Likelihood
    from bilby.core.prior import PriorDict


class BilbyModel(Model):
    """Model class to wrap a bilby likelihood and prior dictionary.

    Parameters
    ----------
    priors :
        Bilby PriorDict object
    likelihood :
        Bilby Likelihood object
    use_ratio : bool
        Whether to use the log-likelihood ratio (True) or log-likelihood
        (False).
    """

    def __init__(
        self,
        *,
        priors: "PriorDict",
        likelihood: "Likelihood",
        use_ratio: bool = False,
    ):
        from bilby.core.prior import PriorDict

        if not isinstance(priors, PriorDict):
            raise TypeError("priors must be an instance of PriorDict")

        self.bilby_priors = priors
        self.bilby_likelihood = likelihood
        self.use_ratio = use_ratio
        self.names = self.bilby_priors.non_fixed_keys
        self._update_bounds()
        self.fixed_parameters = self._fixed_parameters_dict(self.bilby_priors)

        if self.use_ratio:
            self.bilby_log_likelihood_fn = (
                self.bilby_likelihood.log_likelihood_ratio
            )
        else:
            self.bilby_log_likelihood_fn = self.bilby_likelihood.log_likelihood

        self.validate_bilby_likelihood()

    @staticmethod
    def _fixed_parameters_dict(bilby_priors: "PriorDict") -> dict:
        """Dictionary of fixed parameters."""
        theta = {}
        for key in bilby_priors.fixed_keys:
            theta[key] = bilby_priors[key].value
        return theta

    def _update_bounds(self):
        self.bounds = {
            key: [
                self.bilby_priors[key].minimum,
                self.bilby_priors[key].maximum,
            ]
            for key in self.names
        }

    def _try_bilby_likelihood(self, theta):
        """Try to evaluate the bilby likelihood with the given parameters."""
        try:
            return self.bilby_log_likelihood_fn(theta)
        except TypeError:
            self.bilby_likelihood.parameters.update(theta)
            return self.bilby_log_likelihood_fn()

    def validate_bilby_likelihood(self) -> None:
        """Validate the bilby likelihood object"""
        theta = self.fixed_parameters.copy()
        theta_new = self.bilby_priors.sample()
        theta.update(theta_new)
        return self._try_bilby_likelihood(theta)

    def log_likelihood(self, x):
        """Compute the log likelihood"""
        theta = self.fixed_parameters.copy()
        for n in self.names:
            theta[n] = x[n].item()
        return self._try_bilby_likelihood(theta)

    def log_prior(self, x):
        """Compute the log prior.

        Also evaluates the likelihood constraints.
        """
        theta = {n: x[n] for n in self.names}
        return self.bilby_priors.ln_prob(theta, axis=0)

    def new_point(self, N=1):
        """Draw a point from the prior"""
        prior_samples = self.bilby_priors.sample(size=N)
        samples = {n: prior_samples[n] for n in self.names}
        return dict_to_live_points(samples)

    def new_point_log_prob(self, x):
        """Proposal probability for new the point"""
        return self.log_prior(x)

    def from_unit_hypercube(self, x):
        """Map samples from the unit hypercube to the prior."""
        theta = x.copy()
        for n in self.names:
            theta[n] = self.bilby_priors[n].rescale(x[n])
        return theta

    def to_unit_hypercube(self, x):
        """Map samples from the prior to the unit hypercube."""
        theta = x.copy()
        for n in self.names:
            theta[n] = self.bilby_priors[n].cdf(x[n])
        return theta


class BilbyModelLikelihoodConstraint(BilbyModel):
    """Bilby model where prior constraints are included in the likelihood."""

    def log_likelihood(self, x):
        """Compute the log likelihood.

        Also evaluates the likelihood constraints.
        """
        theta = self.fixed_parameters.copy()
        for n in self.names:
            theta[n] = x[n].item()
        if not self.bilby_priors.evaluate_constraints(theta):
            return -np.inf
        return self._try_bilby_likelihood(theta)

    def log_prior(self, x):
        """Compute the log prior."""
        theta = {n: x[n] for n in self.names}
        return self.bilby_priors.ln_prob(theta, axis=0)
