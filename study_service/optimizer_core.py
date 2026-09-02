"""Pairwise Bayesian optimisation core for the OptiCarVis study.

The model observes only pairwise choices.  A PairwiseGP estimates one latent
preference utility, and analytic EUBO selects both configurations in each
adaptive comparison.  The palette is one hot encoded and all sixteen ordered
palette pairs are enumerated while the six continuous values for the two
options are optimised.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import preference_data
import space


def _dependencies():
    try:
        import torch
        from botorch.acquisition.acquisition import AcquisitionFunction
        from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
        from botorch.fit import fit_gpytorch_mll
        from botorch.models.pairwise_gp import (
            PairwiseGP,
            PairwiseLaplaceMarginalLogLikelihood,
        )
        from botorch.models.transforms.input import Normalize
        from botorch.optim import optimize_acqf
    except ImportError as exc:
        raise RuntimeError(
            "Preference optimisation dependencies are missing. Install "
            "study_service/requirements.txt in the study service environment."
        ) from exc
    return {
        "torch": torch,
        "AcquisitionFunction": AcquisitionFunction,
        "EUBO": AnalyticExpectedUtilityOfBestOption,
        "fit": fit_gpytorch_mll,
        "PairwiseGP": PairwiseGP,
        "PairwiseMLL": PairwiseLaplaceMarginalLogLikelihood,
        "Normalize": Normalize,
        "optimize_acqf": optimize_acqf,
    }


def fit_preference_model(training: preference_data.TrainingData):
    if not training.comparisons:
        raise ValueError("at least one completed comparison is required")
    space.validate_model_rows(training.model_rows)
    deps = _dependencies()
    torch = deps["torch"]
    train_x = torch.tensor(training.model_rows, dtype=torch.double)
    train_comp = torch.tensor(training.comparisons, dtype=torch.long)
    bounds = torch.stack(
        [
            torch.zeros(space.D_MODEL, dtype=torch.double),
            torch.ones(space.D_MODEL, dtype=torch.double),
        ]
    )
    model = deps["PairwiseGP"](
        train_x,
        train_comp,
        input_transform=deps["Normalize"](d=space.D_MODEL, bounds=bounds),
    )
    mll = deps["PairwiseMLL"](model.likelihood, model)
    deps["fit"](mll)
    return model


def sobol_pair(comparison_step: int, seed: int = 0):
    """Return the deterministic Sobol pair for a one based comparison step."""
    if comparison_step < 1:
        raise ValueError("comparison_step must be at least 1")
    deps = _dependencies()
    torch = deps["torch"]
    engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
    samples = engine.draw(comparison_step * 2, dtype=torch.double)[-2:]
    options = []
    for row in samples:
        palette_index = min(int(row[3].item() * len(space.PALETTE_VALUES)), 3)
        options.append(
            space.config_from_unit_continuous(
                row[: space.D_CONTINUOUS].tolist(),
                space.PALETTE_VALUES[palette_index],
            )
        )
    if space.config_key(options[0]) == space.config_key(options[1]):
        return sobol_pair(comparison_step, seed=seed + 1)
    return options[0], options[1]


def _fallback_unseen_pair(
    comparison_step: int,
    observed_pair_keys: Iterable,
    seed: int,
):
    observed = set(observed_pair_keys)
    for attempt in range(128):
        pair = sobol_pair(comparison_step + attempt + 1, seed=seed + attempt)
        if space.unordered_pair_key(*pair) not in observed:
            return pair
    raise RuntimeError("could not find an unseen fallback comparison pair")


def propose_eubo_pair(
    model,
    observed_pair_keys: Iterable = (),
    comparison_step: int = space.N_EXPLORATION_COMPARISONS + 1,
    seed: int = 0,
    num_restarts: int = 8,
    raw_samples: int = 256,
):
    """Optimise EUBO for two options while enumerating palette pairs."""
    deps = _dependencies()
    torch = deps["torch"]
    base_acquisition = deps["EUBO"](pref_model=model)
    AcquisitionFunction = deps["AcquisitionFunction"]

    class FixedPalettePairAcquisition(AcquisitionFunction):
        def __init__(self, palette_a: int, palette_b: int):
            super().__init__(model=model)
            self.palette_a = palette_a
            self.palette_b = palette_b

        def forward(self, x):
            # optimize_acqf uses q=1 over a six dimensional auxiliary point.
            # Split it into the continuous dimensions for EUBO's two options.
            packed = x.squeeze(dim=-2)
            cont_a = packed[..., : space.D_CONTINUOUS]
            cont_b = packed[..., space.D_CONTINUOUS :]
            shape = (*cont_a.shape[:-1], len(space.PALETTE_VALUES))
            one_hot_a = torch.zeros(shape, dtype=cont_a.dtype, device=cont_a.device)
            one_hot_b = torch.zeros(shape, dtype=cont_b.dtype, device=cont_b.device)
            one_hot_a[..., self.palette_a] = 1.0
            one_hot_b[..., self.palette_b] = 1.0
            option_a = torch.cat([cont_a, one_hot_a], dim=-1)
            option_b = torch.cat([cont_b, one_hot_b], dim=-1)
            return base_acquisition(torch.stack([option_a, option_b], dim=-2))

    bounds = torch.stack(
        [
            torch.zeros(2 * space.D_CONTINUOUS, dtype=torch.double),
            torch.ones(2 * space.D_CONTINUOUS, dtype=torch.double),
        ]
    )
    best = None
    for palette_a in space.PALETTE_VALUES:
        for palette_b in space.PALETTE_VALUES:
            torch.manual_seed(seed + 17 * palette_a + palette_b)
            packed, value = deps["optimize_acqf"](
                acq_function=FixedPalettePairAcquisition(palette_a, palette_b),
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 8, "maxiter": 200},
            )
            score = float(value.max().detach().cpu())
            row = packed.detach().cpu().squeeze(0).tolist()
            pair = (
                space.config_from_unit_continuous(
                    row[: space.D_CONTINUOUS], palette_a
                ),
                space.config_from_unit_continuous(
                    row[space.D_CONTINUOUS :], palette_b
                ),
            )
            if best is None or score > best[0]:
                best = (score, pair)

    assert best is not None
    pair = best[1]
    pair_key = space.unordered_pair_key(*pair)
    if (
        space.config_key(pair[0]) == space.config_key(pair[1])
        or pair_key in set(observed_pair_keys)
    ):
        return _fallback_unseen_pair(
            comparison_step=comparison_step,
            observed_pair_keys=observed_pair_keys,
            seed=seed + 10_000,
        )
    return pair


def select_best_observed(
    model,
    raw_configs: Sequence[Mapping[str, object]],
    model_rows: Sequence[Sequence[float]],
):
    """Select the evaluated configuration with highest posterior mean utility."""
    if not raw_configs or len(raw_configs) != len(model_rows):
        raise ValueError("raw_configs and model_rows must have equal nonzero length")
    deps = _dependencies()
    torch = deps["torch"]
    x = torch.tensor(model_rows, dtype=torch.double)
    with torch.no_grad():
        means = model.posterior(x).mean.reshape(-1)
    index = int(torch.argmax(means).item())
    return space.validate_config(raw_configs[index]), float(means[index].cpu())
