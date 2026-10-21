"""
Reproducibility utilities (seeding) for the TR-TAA experiments.

The generation and evaluation scripts are stochastic:

  * TR-TAA samples K random composed transforms at every PGD step
    (blur sigma, resize scale, noise sigma, brightness/contrast factor,
    rotation angle, JPEG quality are all drawn uniformly at random in
    src/eot_attack.py).
  * Some evaluation transforms are stochastic (additive Gaussian noise;
    the print-scan pipeline).

Because of this, re-running with a different seed yields a *different*
adversarial perturbation and slightly different metrics; this variance
is quantified by the seed-stability experiment (statistical_analysis.py).

`set_seed` fixes every RNG the pipeline touches so that (a) a single run
is exactly reproducible given its seed, and (b) multiple runs with
different seeds estimate the run-to-run variance.

Usage
-----
    from src.reproducibility import set_seed
    set_seed(args.seed)              # call ONCE, right after parsing args,
                                     # before building datasets/models.
"""

import os
import random

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:            # torch not importable (e.g. stats-only host)
    _HAS_TORCH = False


def set_seed(seed: int, deterministic: bool = True) -> int:
    """
    Seed all RNGs used by the pipeline.

    Args:
        seed: the integer seed.
        deterministic: if True, also request deterministic cuDNN kernels.
            This makes convolutions reproducible at a small speed cost.
            Set to False if a specific op raises a "no deterministic
            implementation" error.

    Returns:
        The seed (echoed back for logging convenience).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Best-effort: some torch versions expose this global switch.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

    return seed
