from contextlib import contextmanager
from collections import OrderedDict

import numpy as np

class BasePolicy:
    def __init__(self):
        self._deterministic = False

    def reset(self):
        """Reset and clean the policy."""
        raise NotImplementedError

    def actions(self, conditions):
        """Compute (symbolic) actions given conditions (observations)"""
        raise NotImplementedError

    def log_pis(self, conditions, actions):
        """Compute (symbolic) log probs for given observations and actions."""
        raise NotImplementedError

    def actions_np(self, conditions):
        """Compute (numeric) actions given conditions (observations)"""
        raise NotImplementedError

    def log_pis_np(self, conditions, actions):
        """Compute (numeric) log probs for given observations and actions."""
        raise NotImplementedError

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.

        Arguments:
            conditions: Observations to run the diagnostics for.
        Returns:
            diagnostics: OrderedDict of diagnostic information.
        """
        diagnostics = OrderedDict({})
        return diagnostics