# GLMM interpretation — pass/fail confirmatory analysis

## Primary inferential model

- **Method:** BinomialBayesMixedGLM_VB
- **Formula:** `pass_fail ~ C(model_id) + C(prompt_variant_id) + temperature + (1|task) + (1|run)`
- **Converged:** True
- **Valid for inference:** True
- **Observations:** 39360
- **Tasks:** 164
- **Runs:** 5

## Random-effect variances

- `task`: 3.10451
- `run`: 0.0588519

## Reference categories

- `model_id` → `deepseek_coder_v2_lite`

## Fixed effects (odds ratios, approximate 95% variational intervals)

- `prompt_variant_id` → `explicit_reasoning`

## Fixed effects (odds ratios, approximate 95% variational intervals)

- `Intercept`: OR=3.954 [3.829, 4.083] (approx. variational posterior interval on the OR scale)
- `C(model_id)[T.llama31_8b]`: OR=0.361 [0.338, 0.385] (approx. variational posterior interval on the OR scale)
- `C(model_id)[T.qwen25_coder_14b]`: OR=6.349 [5.729, 7.036] (approx. variational posterior interval on the OR scale)
- `C(model_id)[T.qwen25_coder_32b]`: OR=5.454 [4.941, 6.020] (approx. variational posterior interval on the OR scale)
- `C(model_id)[T.qwen25_coder_7b]`: OR=1.840 [1.703, 1.988] (approx. variational posterior interval on the OR scale)
- `C(model_id)[T.qwen3_32b]`: OR=0.022 [0.020, 0.023] (approx. variational posterior interval on the OR scale)
- `C(prompt_variant_id)[T.minimal]`: OR=0.971 [0.911, 1.036] (approx. variational posterior interval on the OR scale)
- `C(prompt_variant_id)[T.professional]`: OR=0.983 [0.922, 1.049] (approx. variational posterior interval on the OR scale)
- `C(prompt_variant_id)[T.testing_oriented]`: OR=0.981 [0.920, 1.046] (approx. variational posterior interval on the OR scale)
- `temperature`: OR=1.020 [0.813, 1.281] (approx. variational posterior interval on the OR scale)

## Reduced-model comparison

- Reduced model was not required.

## Sensitivity analysis

The linear probability mixed model was fitted only as a sensitivity analysis.

The fit exhibited singular covariance and a non-positive definite Hessian.

Consequently, its estimates are not used for inference and are omitted from all publication tables.

Representative warnings:

- Random effects covariance is singular.
- Hessian matrix not positive definite.

## Limitations

- Primary inference uses a binomial model on pass/fail; ANOVA on continuous scores remains descriptive.
- Odds ratios describe association on the logit scale; uncertainty should be read from approximate variational posterior intervals (normal approximation on log-odds), not classical frequentist confidence intervals.
- p-values are reported when available but are not treated as the sole evidence of importance.
- Crossed random effects may be weakly identified even when the reduced model converges.
