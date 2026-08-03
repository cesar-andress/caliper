"""Tests for binomial GLMM primary inference pipeline."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from caliper.statistics.glmm_analysis import (
    SENSITIVITY_LPM_LABEL,
    fit_linear_probability_sensitivity,
    run_pass_fail_glmm_analysis,
    validate_binary_outcome,
)


def _binary_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    models = [f"m{i}" for i in range(3)]
    prompts = ["p0", "p1"]
    tasks = [f"t{i}" for i in range(5)]
    rows = []
    for _ in range(n):
        model = rng.choice(models)
        prompt = rng.choice(prompts)
        task = rng.choice(tasks)
        run = int(rng.integers(0, 3))
        temp = float(rng.choice([0.0, 0.2]))
        logit = (-0.5 if model == "m0" else 0.2) + (0.1 if prompt == "p1" else 0.0) + 0.05 * temp
        prob = 1 / (1 + np.exp(-logit))
        rows.append(
            {
                "model_id": model,
                "prompt_variant_id": prompt,
                "task_id": task,
                "run_index": run,
                "temperature": temp,
                "pass_at_1": float(rng.random() < prob),
            }
        )
    return pd.DataFrame(rows)


class TestBinaryValidation:
    def test_binary_outcome_validation(self) -> None:
        validate_binary_outcome(pd.Series([0, 1, 1, 0]))
        with pytest.raises(ValueError, match="must be binary"):
            validate_binary_outcome(pd.Series([0.0, 0.5, 1.0]))


class TestBinomialGLMMHierarchy:
    def test_converged_binomial_glmm_selected_as_primary(self) -> None:
        frame = _binary_frame()
        mock_fit = MagicMock()
        mock_fit.params = np.array([-0.2, 0.4])
        mock_fit.cov_params.return_value = pd.Series([0.01, 0.0225])
        mock_fit.llf = -50.0
        mock_fit.vcp_mean = np.array([0.5, 0.1])

        mock_model = MagicMock()
        mock_model.exog_names = ["Intercept", "C(model_id)[T.m1]"]
        mock_model.fit_vb.return_value = mock_fit

        with patch(
            "statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula",
            return_value=mock_model,
        ):
            result = run_pass_fail_glmm_analysis(frame, metric="pass_at_1")

        assert result.primary.method == "BinomialBayesMixedGLM_VB"
        assert result.primary.valid_for_inference
        assert not result.coefficients.empty
        assert "odds_ratio" in result.coefficients.columns

    def test_zero_run_variance_triggers_reduced_model(self) -> None:
        frame = _binary_frame()
        full_fit = MagicMock()
        full_fit.params = np.array([0.1])
        full_fit.cov_params.return_value = pd.Series([0.04])
        full_fit.vcp_mean = np.array([0.5, -20.0])
        full_fit.llf = -40.0

        reduced_fit = MagicMock()
        reduced_fit.params = np.array([0.1])
        reduced_fit.cov_params.return_value = pd.Series([0.04])
        reduced_fit.vcp_mean = np.array([0.5])
        reduced_fit.llf = -39.0

        mock_model = MagicMock()
        mock_model.exog_names = ["Intercept"]
        mock_model.fit_vb.side_effect = [full_fit, reduced_fit]

        with patch(
            "statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula",
            return_value=mock_model,
        ):
            result = run_pass_fail_glmm_analysis(frame, metric="pass_at_1")

        assert result.reduced_model_needed
        assert result.reduced_model is not None

    def test_cluster_robust_glm_fallback(self) -> None:
        frame = _binary_frame()
        with patch(
            "statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula",
            side_effect=RuntimeError("bayes failed"),
        ):
            result = run_pass_fail_glmm_analysis(frame, metric="pass_at_1")

        assert result.primary.method == "BinomialGLM_cluster_robust_by_task"
        assert any("Fallback" in note for note in result.primary.notes)

    def test_warning_capture_recorded_in_diagnostics(self) -> None:
        frame = _binary_frame()
        mock_fit = MagicMock()
        mock_fit.params = np.array([0.0])
        mock_fit.cov_params.return_value = pd.Series([0.01])
        mock_fit.llf = -10.0
        mock_fit.vcp_mean = np.array([0.3])

        def fit_vb_with_warning() -> MagicMock:
            warnings.warn("Random effects covariance is singular", RuntimeWarning)
            return mock_fit

        mock_model = MagicMock()
        mock_model.exog_names = ["Intercept"]
        mock_model.fit_vb.side_effect = fit_vb_with_warning

        with patch(
            "statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula",
            return_value=mock_model,
        ):
            result = run_pass_fail_glmm_analysis(frame, metric="pass_at_1")

        warning_text = result.diagnostics["warnings"].astype(str).str.cat(sep=" ")
        assert "singular" in warning_text.lower() or "RuntimeWarning" in warning_text

    def test_no_gaussian_mixedlm_as_primary(self) -> None:
        frame = _binary_frame()
        mock_fit = MagicMock()
        mock_fit.params = np.array([0.0])
        mock_fit.cov_params.return_value = pd.Series([0.01])
        mock_fit.llf = -10.0
        mock_fit.vcp_mean = np.array([0.3])

        mock_model = MagicMock()
        mock_model.exog_names = ["Intercept"]
        mock_model.fit_vb.return_value = mock_fit

        with patch(
            "statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula",
            return_value=mock_model,
        ):
            result = run_pass_fail_glmm_analysis(frame, metric="pass_at_1")

        assert "MixedLM" not in result.primary.method


class TestSensitivityMixedLM:
    def test_non_convergent_lpm_excluded_from_publication(self) -> None:
        frame = _binary_frame()
        frame["pass_fail"] = (frame["pass_at_1"] >= 0.5).astype(int)
        mock_result = MagicMock()
        mock_result.converged = False
        mock_result.fe_params.index = pd.Index(["Intercept"])
        mock_result.fe_params.values = np.array([0.2])
        mock_result.bse_fe.values = np.array([0.1])
        mock_result.pvalues.values = np.array([0.05])
        mock_result.conf_int.return_value = pd.DataFrame([[0.0, 0.4]])
        mock_result.cov_re = pd.DataFrame([[0.01]])
        mock_result.llf = -1.0
        mock_result.aic = 2.0
        mock_result.bic = 3.0
        mock_result.hessian = np.array([[1.0]])

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result

        with patch("statsmodels.formula.api.mixedlm", return_value=mock_model):
            diag, coef = fit_linear_probability_sensitivity(frame)

        assert diag.method == SENSITIVITY_LPM_LABEL
        assert not diag.valid_for_inference
        assert coef["include_in_publication"].eq(False).all()
