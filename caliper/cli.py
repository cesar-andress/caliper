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


@main.group()
def analyze() -> None:
    """Post-hoc statistical analysis on saved results."""


@analyze.command("variance")
@click.option("--results", "-r", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default=None, help="Filter to one metric name.")
def analyze_variance(results: Path, metric: str | None) -> None:
    """Decompose score variance by experimental factor (Paper 1)."""
    import pandas as pd

    from caliper.statistics.descriptive import descriptive_by_factor
    from caliper.statistics.gtheory import estimate_g_variance_components
    from caliper.statistics.prepare import prepare_results_table
    from caliper.statistics.variance import decompose_variance

    suffix = results.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(results)
    elif suffix == ".jsonl":
        raw = pd.read_json(results, orient="records", lines=True)
    else:
        raw = pd.read_csv(results)

    df = prepare_results_table(raw, metric_name=metric)
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
@click.option("--metric", default=None, help="Filter to one metric name.")
@click.option("--effect-size", default=0.05, show_default=True, type=float)
@click.option("--simulations", default=300, show_default=True, type=int)
def analyze_power(results: Path, metric: str | None, effect_size: float, simulations: int) -> None:
    """Simulate statistical power for Paper 1 designs."""
    import pandas as pd

    from caliper.statistics.gtheory import estimate_g_variance_components
    from caliper.statistics.power_sim import simulate_power_grid
    from caliper.statistics.prepare import prepare_results_table

    suffix = results.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(results)
    elif suffix == ".jsonl":
        raw = pd.read_json(results, orient="records", lines=True)
    else:
        raw = pd.read_csv(results)

    df = prepare_results_table(raw, metric_name=metric)
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
@click.option("--metric", "-m", default=None, help="Metric name to analyze.")
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
    n_bootstrap: int,
    seed: int,
    output_dir: Path,
    reports_dir: Path | None,
) -> None:
    """Quantify ranking fragility under task/run/prompt bootstrap (Paper 2)."""
    from caliper.ranking import run_ranking_fragility_from_file

    if reports_dir is None:
        reports_dir = output_dir / "plots"

    outputs = run_ranking_fragility_from_file(
        results_path,
        metric_name=metric,
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
