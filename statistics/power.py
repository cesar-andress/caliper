"""Statistical power analysis for LLM evaluation experiments (Paper 1)."""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class PowerAnalysisResult:
    """Result of a statistical power computation."""

    effect_size: float
    alpha: float
    n_per_group: int
    power: float
    test: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "n_per_group": self.n_per_group,
            "power": self.power,
            "test": self.test,
        }


def compute_power(
    effect_size: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
    test: str = "t-test",
) -> PowerAnalysisResult:
    """Compute statistical power for detecting a given effect size.

    Uses the non-central t-distribution approximation for two-sample t-tests.

    Args:
        effect_size: Cohen's d (standardized mean difference).
        n_per_group: Sample size per group.
        alpha: Significance level.
        test: Test type identifier (currently only ``t-test``).

    Returns:
        PowerAnalysisResult with estimated power.
    """
    if test != "t-test":
        msg = f"Unsupported test: {test}"
        raise ValueError(msg)

    df = 2 * n_per_group - 2
    nct = stats.nct(df, effect_size * (n_per_group / 2) ** 0.5)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = float(1 - nct.cdf(t_crit) + nct.cdf(-t_crit))

    return PowerAnalysisResult(
        effect_size=effect_size,
        alpha=alpha,
        n_per_group=n_per_group,
        power=power,
        test=test,
    )
