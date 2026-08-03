"""CALIPER command-line interface."""

from __future__ import annotations

from pathlib import Path

import click
import structlog

from caliper import __version__
from caliper.config.errors import ConfigValidationError
from caliper.config.loader import format_config_summary, load_config, validate_config
from caliper.runners.experiment import ExperimentRunner

logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="caliper")
def main() -> None:
    """CALIPER — variance, power, and ranking fragility in LLM evaluations."""


@main.command()
@click.argument(
    "config_arg",
    required=False,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--config", "-c",
    "config_opt",
    type=click.Path(exists=True, path_type=Path),
    help="Path to experiment YAML config.",
)
@click.option("--dry-run", is_flag=True, help="Validate config and plan without executing.")
@click.option(
    "--resume",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Resume a previous run from its output directory.",
)
def run(
    config_arg: Path | None,
    config_opt: Path | None,
    dry_run: bool,
    resume: Path | None,
) -> None:
    """Run an experiment from a YAML configuration file."""
    config_path = config_opt or config_arg
    if config_path is None:
        raise click.UsageError("Missing config path. Pass CONFIG or --config/-c.")

    experiment_config = load_config(config_path)
    runner = ExperimentRunner(
        experiment_config,
        config_path=config_path,
        dry_run=dry_run,
        resume_dir=resume,
    )
    manifest = runner.run()
    click.echo(
        f"Run {manifest.run_id} finished with status: {manifest.status} "
        f"({manifest.completed_cells} completed, "
        f"{manifest.failed_cells} failed, "
        f"{manifest.skipped_cells} skipped / "
        f"{manifest.total_cells} total cells)"
    )


@main.command()
@click.option(
    "--config", "-c", required=True, type=click.Path(exists=True, path_type=Path),
    help="Path to experiment YAML config.",
)
def plan(config: Path) -> None:
    """Print the experiment plan (full factorial design)."""
    experiment_config = load_config(config)
    click.echo(format_config_summary(experiment_config))
    click.echo("")
    combos = ExperimentRunner(experiment_config, dry_run=True).plan_combinations()
    click.echo(f"Combinations ({len(combos)}):")
    for i, combo in enumerate(combos, 1):
        click.echo(f"  [{i}] {combo}")


@main.command()
@click.option(
    "--config", "-c", required=True, type=click.Path(exists=True, path_type=Path),
    help="Path to experiment YAML config.",
)
def validate(config: Path) -> None:
    """Validate an experiment configuration file."""
    errors = validate_config(config)
    if errors:
        raise ConfigValidationError(config.resolve(), errors)

    experiment_config = load_config(config)
    click.echo(f"Config '{experiment_config.experiment_id}' is valid.")
    click.echo(format_config_summary(experiment_config))


@main.command("inspect-missing-cells")
@click.argument(
    "experiment_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the frozen experiment YAML config.",
)
@click.option(
    "--write-retry-config",
    is_flag=True,
    help="Write retry_missing_cells.json alongside the diagnostic report.",
)
def inspect_missing_cells_cmd(
    experiment_dir: Path,
    config: Path,
    write_retry_config: bool,
) -> None:
    """Compare expected factorial cells against checkpoints and results."""
    from caliper.runners.missing_cells import inspect_missing_cells, write_missing_cells_report

    experiment_config = load_config(config)
    report = inspect_missing_cells(
        experiment_dir,
        experiment_config,
        config_dir=config.parent.resolve(),
    )
    outputs = write_missing_cells_report(
        experiment_dir,
        report,
        write_retry_config=write_retry_config,
        config_path=config,
    )
    counts = report["counts"]
    click.echo(
        f"Missing-cell inspection complete for {experiment_config.experiment_id}: "
        f"{counts['missing_cell_ids']} missing / {counts['expected_cells']} expected"
    )
    click.echo(f"  JSON: {outputs['json']}")
    click.echo(f"  Markdown: {outputs['markdown']}")
    if "retry_spec" in outputs:
        click.echo(f"  Retry spec: {outputs['retry_spec']}")


@main.command("retry-missing")
@click.argument(
    "experiment_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--report",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to missing_cells_report.json or retry_missing_cells.json.",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Optional config override; defaults to report config_path.",
)
def retry_missing_cmd(
    experiment_dir: Path,
    report: Path,
    config: Path | None,
) -> None:
    """Retry only missing factorial cells without overwriting completed ones."""
    from caliper.runners.retry_missing import retry_missing_cells

    summary = retry_missing_cells(
        experiment_dir,
        report_path=report,
        config_path=config,
    )
    click.echo(
        f"Recovery run {summary['recovery_run_id']} finished for "
        f"original run {summary['original_run_id']}: "
        f"{summary['recovered_cells']} recovered, "
        f"{summary['still_failed_cells']} still failed, "
        f"{summary['skipped_cells']} skipped"
    )
    click.echo(f"  Remaining missing cells: {summary['remaining_missing_cells']}")
    click.echo(f"  Audit trail: {summary['audit_path']}")


@main.command()
@click.argument(
    "experiment_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--limit", default=5, show_default=True, type=int, help="Number of rows to show.")
def inspect(experiment_dir: Path, limit: int) -> None:
    """Inspect predictions, references, and metric values from a completed experiment."""
    from caliper.evaluation.inspect_output import format_inspection, inspect_experiment, metric_means_from_results
    from caliper.storage.formats import read_results

    records = inspect_experiment(experiment_dir, limit=limit)
    click.echo(format_inspection(records))

    results_path = experiment_dir / "results.parquet"
    if results_path.exists():
        means = metric_means_from_results(read_results(results_path))
        if means:
            click.echo("\nMetric means (all completed cells):")
            for name in sorted(means):
                click.echo(f"  {name}: {means[name]:.4f}")


@main.command()
@click.argument(
    "results_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Experiment YAML config used to generate the results.",
)
@click.option(
    "--enable-code-execution",
    is_flag=True,
    help="Enable test_pass metric (still a placeholder; no code is executed).",
)
@click.option(
    "--enable-llm-judge",
    is_flag=True,
    help="Enable LLM-as-judge metric (placeholder; disabled by default).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for evaluation outputs (defaults to results directory).",
)
def evaluate(
    results_path: Path,
    config: Path,
    enable_code_execution: bool,
    enable_llm_judge: bool,
    output_dir: Path | None,
) -> None:
    """Evaluate saved experiment results with configured metrics."""
    from caliper.evaluation import EvaluationOptions, evaluate_results_file

    experiment_config = load_config(config)
    options = EvaluationOptions(
        enable_code_execution=enable_code_execution,
        enable_llm_judge=enable_llm_judge,
    )
    summary = evaluate_results_file(
        results_path,
        experiment_config,
        config_path=config,
        options=options,
        output_dir=output_dir,
    )
    click.echo(f"Evaluated {summary['rows_evaluated']} result rows.")
    if summary.get("output_parquet"):
        click.echo(f"  Parquet: {summary['output_parquet']}")
    if summary.get("output_jsonl"):
        click.echo(f"  JSONL:   {summary['output_jsonl']}")


@main.command("export-artifact")
@click.argument(
    "experiment_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--force", is_flag=True, help="Replace an existing artifact directory.")
def export_artifact_cmd(experiment_dir: Path, force: bool) -> None:
    """Export a reproducibility artifact bundle for a completed experiment."""
    from caliper.runners.artifact_export import export_artifact

    result = export_artifact(experiment_dir, force=force)
    click.echo(f"Artifact exported to: {result.artifact_dir}")
    click.echo(f"Complete: {result.verification.complete}")
    if result.verification.warnings:
        click.echo("Warnings:")
        for warning in result.verification.warnings:
            click.echo(f"  - {warning}")
    if result.verification.errors:
        click.echo("Errors:")
        for error in result.verification.errors:
            click.echo(f"  - {error}")
        raise SystemExit(1)


@main.command("validate-confirmatory")
@click.option(
    "--benchmark",
    type=click.Choice(["humaneval", "mbpp"], case_sensitive=False),
    default="humaneval",
    show_default=True,
    help="Confirmatory benchmark to validate.",
)
@click.option("--model", default=None, help="Model id from confirmatory config (default: qwen25_coder_7b).")
@click.option(
    "--prompt",
    default=None,
    type=click.Choice(["minimal", "explicit_reasoning", "testing_oriented", "professional"]),
    help="Controlled prompt variant (default: minimal).",
)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--runs", default=1, show_default=True, type=int)
@click.option("--tasks", default=3, show_default=True, type=int, help="Number of benchmark tasks to exercise.")
@click.option("--verbose", is_flag=True, help="Print per-stage validation output.")
@click.option(
    "--reference-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Confirmatory YAML reference (default: 40-task humaneval config).",
)
@click.option(
    "--expected-total-tasks",
    type=int,
    default=None,
    help="Require exact benchmark task count (e.g., 164 for full HumanEval+).",
)
def validate_confirmatory_cmd(
    benchmark: str,
    model: str | None,
    prompt: str | None,
    temperature: float,
    runs: int,
    tasks: int,
    verbose: bool,
    reference_config: Path | None,
    expected_total_tasks: int | None,
) -> None:
    """Run end-to-end pre-flight validation for a confirmatory experiment."""
    from caliper.validation.config_builder import DEFAULT_MODEL, DEFAULT_PROMPT
    from caliper.validation.confirmatory import run_confirmatory_validation

    report = run_confirmatory_validation(
        benchmark=benchmark,
        model=model or DEFAULT_MODEL,
        prompt=prompt or DEFAULT_PROMPT,
        temperature=temperature,
        runs=runs,
        tasks=tasks,
        verbose=verbose,
        reference_config=reference_config,
        expected_total_tasks=expected_total_tasks,
    )

    click.echo(f"Pre-flight validation output: {report.output_dir}")
    click.echo(f"Ready to launch: {'YES' if report.ready_to_launch else 'NO'}")
    click.echo(f"Report: {Path(report.output_dir) / 'validation_report.md'}")
    click.echo(f"Checklist: {Path(report.output_dir) / 'launch_checklist.md'}")

    click.echo("\nStage summary:")
    for row in report.status_table():
        click.echo(f"  [{row['status']}] {row['stage']}: {row['message']}")

    if report.timing.observations:
        click.echo(
            f"\nTiming: {report.timing.per_observation_ms():.0f} ms/observation "
            f"(est. 9,600 cells ≈ {report.timing.estimate_hours(9600):.1f} h)"
        )

    if not report.ready_to_launch:
        click.echo("\nFailures:")
        for failure in report.failures:
            click.echo(f"  - {failure.stage.value}: {failure.root_cause or failure.message}")
            if failure.recommended_fix:
                click.echo(f"    Fix: {failure.recommended_fix}")
        raise SystemExit(1)


@main.group()
def benchmarks() -> None:
    """Official benchmark loaders and confirmatory study preparation."""


@benchmarks.command("materialize")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/benchmarks"),
    show_default=True,
)
@click.option("--benchmark", type=click.Choice(["humaneval_plus", "mbpp", "all"]), default="all")
@click.option("--limit", type=int, default=None, help="Optional cap on tasks written.")
def benchmarks_materialize(output_dir: Path, benchmark: str, limit: int | None) -> None:
    """Download/parse official benchmarks and write CALIPER JSONL datasets."""
    from caliper.benchmarks.materialize import materialize_all, materialize_benchmark

    if benchmark == "all":
        paths = materialize_all(output_dir, limit=limit)
        for name, path in paths.items():
            click.echo(f"  {name}: {path}")
    else:
        path = materialize_benchmark(benchmark, output_dir, limit=limit)  # type: ignore[arg-type]
        click.echo(f"Wrote {path}")


@benchmarks.command("write-configs")
@click.option(
    "--configs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("configs/paper1"),
    show_default=True,
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/benchmarks"),
    show_default=True,
)
@click.option("--task-subset-size", type=int, default=40, show_default=True)
@click.option("--seed", type=int, default=20260404, show_default=True)
def benchmarks_write_configs(
    configs_dir: Path,
    data_dir: Path,
    task_subset_size: int,
    seed: int,
) -> None:
    """Generate Paper 1 confirmatory YAML configs (HumanEval+ and MBPP)."""
    from caliper.benchmarks.experiment_yaml import expected_cell_count, write_confirmatory_configs

    paths = write_confirmatory_configs(
        configs_dir=configs_dir,
        data_dir=data_dir,
        task_subset_size=task_subset_size,
        seed=seed,
    )
    cells = expected_cell_count(task_subset_size)
    for name, path in paths.items():
        click.echo(f"  {name}: {path} ({cells} expected cells)")
    click.echo(
        "Confirmatory configs ready. Run `caliper validate -c <config>` before launching."
    )


@benchmarks.command("write-humaneval-full-config")
@click.option(
    "--reference-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/paper1/confirmatory_humaneval.yaml"),
    show_default=True,
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/benchmarks/humaneval_plus.jsonl"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("configs/paper1/confirmatory_humaneval_full.yaml"),
    show_default=True,
)
def benchmarks_write_humaneval_full_config(
    reference_config: Path,
    dataset: Path,
    output: Path,
) -> None:
    """Generate the 164-task HumanEval+ confirmatory extension config."""
    from caliper.benchmarks.experiment_yaml import expected_cell_count, write_humaneval_full_config
    from caliper.validation.protocol_comparison import assert_protocol_equivalent_except_tasks

    path = write_humaneval_full_config(
        reference_path=reference_config,
        dataset_path=dataset,
        output_path=output,
    )
    result = assert_protocol_equivalent_except_tasks(
        subset_path=reference_config,
        full_path=path,
    )
    click.echo(f"Wrote {path} ({result.full_task_count} tasks, {result.expected_full_cells} cells)")
    click.echo("Protocol comparison: PASS")


@main.command("compare-protocol")
@click.option(
    "--subset-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/paper1/confirmatory_humaneval.yaml"),
    show_default=True,
)
@click.option(
    "--full-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/paper1/confirmatory_humaneval_full.yaml"),
    show_default=True,
)
@click.option(
    "--report",
    type=click.Path(path_type=Path),
    default=Path("docs/paper1_humaneval_full_protocol_comparison.md"),
    show_default=True,
)
def compare_protocol_cmd(subset_config: Path, full_config: Path, report: Path) -> None:
    """Compare 40-task and 164-task confirmatory protocols; fail on unintended diffs."""
    from caliper.validation.protocol_comparison import write_protocol_comparison_report

    result = write_protocol_comparison_report(
        report,
        subset_path=subset_config,
        full_path=full_config,
    )
    click.echo(f"Protocol comparison report: {report}")
    click.echo(f"Result: {'PASS' if result.passed else 'FAIL'}")
    if not result.passed:
        for diff in result.differences:
            click.echo(f"  - {diff}")
        raise SystemExit(1)


@main.command("experiment-status")
@click.argument(
    "experiment_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Experiment YAML config for expected cell count.",
)
def experiment_status_cmd(experiment_dir: Path, config: Path | None) -> None:
    """Report progress, throughput, and ETA for a factorial experiment."""
    from caliper.runners.experiment_status import collect_experiment_status, format_experiment_status

    status = collect_experiment_status(experiment_dir, config_path=config)
    click.echo(format_experiment_status(status))


@main.group()
def ollama() -> None:
    """Inspect and manage local Ollama models."""


@ollama.command("list")
@click.option(
    "--base-url",
    default="http://localhost:11434",
    show_default=True,
    help="Ollama HTTP base URL.",
)
def ollama_list(base_url: str) -> None:
    """List models available in the local Ollama instance."""
    from caliper.models.ollama_client import OllamaConnectionError
    from caliper.models.ollama_provider import list_local_models

    try:
        models = list_local_models(base_url=base_url)
    except OllamaConnectionError as exc:
        raise click.ClickException(
            f"Could not reach Ollama at {base_url}. Is Ollama running? ({exc})"
        ) from exc

    if not models:
        click.echo(f"No models found at {base_url}.")
        return

    click.echo(f"Models at {base_url}:")
    for name in models:
        click.echo(f"  - {name}")


@main.group()
def analyze() -> None:
    """Post-hoc statistical analysis on saved results."""


@analyze.command("variance")
@click.option("--results", "-r", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default=None, help="Metric name (defaults to config primary_metric).")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Experiment YAML (defaults to config.yaml beside results).",
)
def analyze_variance(results: Path, metric: str | None, config: Path | None) -> None:
    """Decompose score variance by experimental factor (Paper 1)."""
    import pandas as pd

    from caliper.config.metrics import resolve_analysis_metric
    from caliper.statistics.descriptive import descriptive_by_factor
    from caliper.statistics.gtheory import estimate_g_variance_components
    from caliper.statistics.prepare import prepare_results_table
    from caliper.statistics.variance import decompose_variance

    resolved_metric, warnings = resolve_analysis_metric(
        metric=metric,
        results_path=results,
        config_path=config,
    )
    for warning in warnings:
        click.echo(f"Warning: {warning}", err=True)

    suffix = results.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(results)
    elif suffix == ".jsonl":
        raw = pd.read_json(results, orient="records", lines=True)
    else:
        raw = pd.read_csv(results)

    df = prepare_results_table(raw, metric_name=resolved_metric)
    if resolved_metric is not None:
        click.echo(f"Using metric: {resolved_metric}")
    click.echo("Variance decomposition (sequential ANOVA):")
    components = decompose_variance(df)
    for key, value in components.as_dict().items():
        if isinstance(value, float):
            click.echo(f"  {key}: {value:.6f}")
        else:
            click.echo(f"  {key}: {value}")

    click.echo("\nDescriptive stats by model:")
    if "model" in df.columns:
        desc = descriptive_by_factor(df, "model")
        click.echo(desc.to_string(index=False))

    click.echo("\nG-study components:")
    gstudy = estimate_g_variance_components(df)
    for facet, var in gstudy.components.items():
        click.echo(f"  {facet}: {var:.6f}")


@analyze.command("power")
@click.option("--results", "-r", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default=None, help="Metric name (defaults to config primary_metric).")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Experiment YAML (defaults to config.yaml beside results).",
)
@click.option("--effect-size", default=0.05, show_default=True, type=float)
@click.option("--simulations", default=300, show_default=True, type=int)
def analyze_power(
    results: Path,
    metric: str | None,
    config: Path | None,
    effect_size: float,
    simulations: int,
) -> None:
    """Simulate statistical power for Paper 1 designs."""
    import pandas as pd

    from caliper.config.metrics import resolve_analysis_metric
    from caliper.statistics.gtheory import estimate_g_variance_components
    from caliper.statistics.power_sim import simulate_power_grid
    from caliper.statistics.prepare import prepare_results_table

    resolved_metric, warnings = resolve_analysis_metric(
        metric=metric,
        results_path=results,
        config_path=config,
    )
    for warning in warnings:
        click.echo(f"Warning: {warning}", err=True)

    suffix = results.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(results)
    elif suffix == ".jsonl":
        raw = pd.read_json(results, orient="records", lines=True)
    else:
        raw = pd.read_csv(results)

    df = prepare_results_table(raw, metric_name=resolved_metric)
    if resolved_metric is not None:
        click.echo(f"Using metric: {resolved_metric}")
    components = estimate_g_variance_components(df).components
    grid = simulate_power_grid(
        components,
        effect_size=effect_size,
        task_counts=[3, 5, 10],
        prompt_counts=[1, 2, 3],
        run_counts=[1, 3, 5],
        n_simulations=simulations,
    )
    click.echo(f"Power simulation (effect_size={effect_size}):")
    click.echo(grid.to_string(index=False))


@main.command("ranking-fragility")
@click.argument(
    "results_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--metric", "-m", default=None, help="Metric name (defaults to config primary_metric).")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Experiment YAML (defaults to config.yaml beside results).",
)
@click.option("--n-bootstrap", default=500, show_default=True, type=int, help="Bootstrap iterations per facet.")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("reports/ranking_fragility"),
    help="Directory for summary CSV and bootstrap samples.",
)
@click.option(
    "--reports-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for plots (defaults to output-dir/plots).",
)
def ranking_fragility_cmd(
    results_path: Path,
    metric: str | None,
    config: Path | None,
    n_bootstrap: int,
    seed: int,
    output_dir: Path,
    reports_dir: Path | None,
) -> None:
    """Quantify ranking fragility under task/run/prompt bootstrap (Paper 2)."""
    from caliper.config.metrics import resolve_analysis_metric
    from caliper.ranking import run_ranking_fragility_from_file

    resolved_metric, warnings = resolve_analysis_metric(
        metric=metric,
        results_path=results_path,
        config_path=config,
    )
    for warning in warnings:
        click.echo(f"Warning: {warning}", err=True)
    if resolved_metric is not None:
        click.echo(f"Using metric: {resolved_metric}")

    if reports_dir is None:
        reports_dir = output_dir / "plots"

    outputs = run_ranking_fragility_from_file(
        results_path,
        metric_name=resolved_metric,
        n_bootstrap=n_bootstrap,
        seed=seed,
        output_dir=output_dir,
        reports_dir=reports_dir,
    )
    click.echo(f"Ranking fragility analysis complete ({len(outputs.baseline_scores)} models).")
    click.echo(f"  Summary:       {output_dir / 'ranking_fragility_summary.csv'}")
    click.echo(f"  Bootstrap:     {output_dir / 'bootstrap_samples.parquet'}")
    click.echo(f"  Rank probs:    {output_dir / 'rank_probabilities.csv'}")
    click.echo(f"  Pairwise:      {output_dir / 'pairwise_reversals.csv'}")
    click.echo(f"  Fragility idx: {outputs.summary['fragility_index'].mean():.4f} (mean)")
    for name, path in outputs.plot_paths.items():
        click.echo(f"  Plot [{name}]: {path}")


@analyze.command("robustness")
@click.option(
    "--experiment-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("experiments/paper1_ollama_pilot"),
    show_default=True,
    help="Completed experiment directory with results/statistical_dataset.",
)
@click.option("--metric", default=None, help="Primary metric (defaults to config primary_metric).")
@click.option("--n-bootstrap", default=5000, show_default=True, type=int, help="Ranking bootstrap iterations.")
@click.option("--fast", is_flag=True, help="Use 500 bootstrap iterations for a quick run.")
def analyze_robustness(
    experiment_dir: Path,
    metric: str | None,
    n_bootstrap: int,
    fast: bool,
) -> None:
    """Run robust ANOVA, convergence, sensitivity, and bootstrap analyses (Paper 1)."""
    from caliper.runners.experiment_paths import resolve_experiment_dir
    from caliper.statistics.robustness_report import run_robustness_analysis

    resolved_dir = resolve_experiment_dir(experiment_dir)
    iterations = 500 if fast else n_bootstrap
    out = run_robustness_analysis(
        resolved_dir,
        metric=metric,
        n_bootstrap=iterations,
    )
    click.echo(f"Resolved experiment directory: {resolved_dir}")
    click.echo(f"Robustness analysis complete: {out}")
    click.echo(f"  Summary: {out / 'summary' / 'robustness_section.md'}")


@analyze.command("task-sampling")
@click.option(
    "--full-experiment",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--subset-experiment",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--metric", default="pass_at_1", show_default=True)
@click.option("--n-subsets", default=1000, show_default=True, type=int)
@click.option("--seed", default=20260404, show_default=True, type=int)
def analyze_task_sampling_cmd(
    full_experiment: Path,
    subset_experiment: Path,
    metric: str,
    n_subsets: int,
    seed: int,
) -> None:
    """Compare 40-task subset stability against the full HumanEval+ benchmark."""
    from caliper.statistics.task_sampling import run_task_sampling_analysis

    paths = run_task_sampling_analysis(
        full_experiment,
        subset_experiment,
        metric=metric,
        n_subsets=n_subsets,
        seed=seed,
    )
    click.echo(f"Task-sampling analysis written to {paths.root}")


@analyze.command("design-guidance")
@click.option(
    "--experiment-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--metric", default="pass_at_1", show_default=True)
def analyze_design_guidance_cmd(experiment_dir: Path, metric: str) -> None:
    """Export actionable D-study design guidance (placeholders until completion)."""
    from caliper.statistics.design_guidance import export_design_guidance

    paths = export_design_guidance(experiment_dir, metric=metric)
    click.echo(f"Design guidance written to {paths.root}")


@analyze.command("fragility")
@click.option("--results", "-r", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--noise-scale", default=0.01, show_default=True, type=float)
@click.option("--n-perturbations", default=1000, show_default=True, type=int)
def analyze_fragility(results: Path, noise_scale: float, n_perturbations: int) -> None:
    """Measure ranking fragility under score perturbation (Paper 2)."""
    import pandas as pd

    from caliper.ranking.fragility import compute_ranking_fragility

    df = pd.read_parquet(results) if results.suffix == ".parquet" else pd.read_csv(results)
    result = compute_ranking_fragility(
        df, noise_scale=noise_scale, n_perturbations=n_perturbations
    )
    for key, value in result.as_dict().items():
        click.echo(f"  {key}: {value}")
