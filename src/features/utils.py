import warnings
from contextlib import contextmanager

@contextmanager
def suppress_runtime_warnings():
    """Context manager to suppress specific runtime warnings during SABR calculations.
    (Safe, isolated; does not change global filters.)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero encountered.*")
        yield
