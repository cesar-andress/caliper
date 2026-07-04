"""Bootstrap confidence intervals for evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap confidence interval for a scalar statistic."""

    statistic: float
    lower: float
    upper: float
    alpha: float
    n_bootstrap: int
    method: str = "percentile"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "statistic": self.statistic,
            "lower": self.lower,
            "upper": self.upper,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "method": self.method,
        }


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    *,
    stat_fn: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Compute a percentile bootstrap confidence interval."""
    if stat_fn is None:
        stat_fn = np.mean

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return BootstrapResult(
            statistic=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            alpha=alpha,
            n_bootstrap=n_bootstrap,
        )

    rng = np.random.default_rng(seed)
    observed = float(stat_fn(arr))
    boot_stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot_stats[i] = stat_fn(sample)

    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)
    lower, upper = np.percentile(boot_stats, [lower_q, upper_q])

    return BootstrapResult(
        statistic=observed,
        lower=float(lower),
        upper=float(upper),
        alpha=alpha,
        n_bootstrap=n_bootstrap,
    )


def bootstrap_ci_by_factor(
    df: pd.DataFrame,
    factor_col: str,
    *,
    value_col: str = "metric_value",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute bootstrap CIs of the mean for each level of a factor."""
    rows: list[dict[str, float | str | int]] = []
    for level, group in df.groupby(factor_col, observed=True):
        level_seed = seed + (hash(str(level)) % 10_000)
        result = bootstrap_ci(
            group[value_col],
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=level_seed,
        )
        rows.append(
            {
                factor_col: level,
                "mean": result.statistic,
                "ci_lower": result.lower,
                "ci_upper": result.upper,
                "n": len(group),
                "n_bootstrap": result.n_bootstrap,
            }
        )
    return pd.DataFrame(rows)
