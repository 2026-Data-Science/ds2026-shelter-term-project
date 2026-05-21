from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Iterable, Optional
import warnings

# KMeans/joblib can emit noisy physical-core detection warnings on some macOS shells.
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.classification.features import LABEL_ORDER, engineer_features
from src.clustering.preprocessing import build_clustering_pipeline


DEFAULT_K_RANGE = (2, 3, 4, 5, 6)
DEFAULT_PCA_VARIANCE = 0.95
DEFAULT_RANDOM_STATE = 42
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 3000
DEFAULT_SPECIES_ORDER = ("Cat", "Dog")


@dataclass
class ClusteringAnalysisResult:
    """Container for the clustering model, evaluation tables, and plot-ready PCA data."""

    segment_name: str
    raw_shape: tuple[int, int]
    preprocessed_shape: tuple[int, int]
    pca_shape: tuple[int, int]
    pca_variance_threshold: float
    pca_explained_variance: float
    pca_first_two_variance: float
    selected_k: int
    inertia: float
    silhouette_sample: float
    top_breeds: list[str]
    candidate_scores: pd.DataFrame
    cluster_summary: pd.DataFrame
    outcome_distribution: pd.DataFrame
    pca_coordinates: pd.DataFrame
    cluster_centers_2d: pd.DataFrame


def _as_dense_array(matrix: object) -> np.ndarray:
    """Convert sklearn outputs to a dense float array before PCA and K-Means."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _top_value(series: pd.Series) -> str:
    """Return the most common category with its within-cluster percentage."""
    cleaned = series.fillna("Unknown").astype(str)
    if cleaned.empty:
        return "Unknown (0.0%)"
    counts = cleaned.value_counts(dropna=False)
    value = counts.index[0]
    ratio = counts.iloc[0] / len(cleaned) * 100.0
    return f"{value} ({ratio:.1f}%)"


def _format_percent(value: float) -> str:
    """Format a numeric ratio as a compact percentage string for reports."""
    return f"{value:.2f}%"


def _choose_best_k(candidate_scores: pd.DataFrame) -> int:
    """Pick the K with the highest sampled silhouette score, preferring smaller K on ties."""
    ordered = candidate_scores.sort_values(
        by=["silhouette_sample", "k"],
        ascending=[False, True],
    )
    return int(ordered.iloc[0]["k"])


def _score_kmeans_candidates(
    reduced: np.ndarray,
    k_range: Iterable[int],
    silhouette_sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Evaluate candidate K values with inertia and sampled silhouette score."""
    rows: list[dict[str, float]] = []
    sample_size: Optional[int]
    if len(reduced) > silhouette_sample_size:
        sample_size = silhouette_sample_size
    else:
        sample_size = None

    for k in k_range:
        if k < 2:
            raise ValueError("KMeans clustering needs k >= 2.")
        if k >= len(reduced):
            raise ValueError("KMeans k must be smaller than the number of rows.")

        # K-Means is fit on PCA-reduced features so distance calculations are less noisy.
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(reduced)
        score = silhouette_score(
            reduced,
            labels,
            sample_size=sample_size,
            random_state=random_state,
        )
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette_sample": float(score),
            }
        )

    return pd.DataFrame(rows)


def _summarize_clusters(
    raw_features: pd.DataFrame,
    labels: np.ndarray,
    y: Optional[pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create human-readable cluster profiles without using labels for model fitting."""
    engineered = engineer_features(raw_features).copy()
    engineered["cluster"] = labels

    rows: list[dict[str, object]] = []
    total = len(engineered)
    for cluster_id in sorted(engineered["cluster"].unique()):
        subset = engineered[engineered["cluster"] == cluster_id]
        count = len(subset)
        median_age_days = float(subset["age_days"].median())
        rows.append(
            {
                "cluster": int(cluster_id),
                "count": int(count),
                "ratio": _format_percent(count / total * 100.0),
                "median_age_days": round(median_age_days, 1),
                "median_age_years": round(median_age_days / 365.0, 2),
                "has_name": _format_percent(float(subset["has_name"].mean()) * 100.0),
                "mixed_breed": _format_percent(
                    float(subset["is_mixed_breed"].mean()) * 100.0
                ),
                "top_animal_type": _top_value(subset["animal_type"]),
                "top_breed": _top_value(subset["primary_breed"]),
                "top_color": _top_value(subset["primary_color"]),
                "top_neuter_status": _top_value(subset["neuter_status"]),
                "top_season": _top_value(subset["outcome_season"]),
            }
        )
    cluster_summary = pd.DataFrame(rows)

    if y is None:
        return cluster_summary, pd.DataFrame()

    # OutcomeType is used only after clustering to interpret groups, never as input.
    outcome_frame = pd.DataFrame(
        {
            "cluster": labels,
            "OutcomeType": y.reset_index(drop=True),
        }
    )
    counts = pd.crosstab(outcome_frame["cluster"], outcome_frame["OutcomeType"])
    label_order = [label for label in LABEL_ORDER if label in counts.columns]
    label_order += sorted(label for label in counts.columns if label not in label_order)
    counts = counts.reindex(columns=label_order, fill_value=0)
    rates = counts.div(counts.sum(axis=1), axis=0) * 100.0

    outcome_rows: list[dict[str, object]] = []
    for cluster_id, row in counts.iterrows():
        dominant = str(row.idxmax())
        result: dict[str, object] = {
            "cluster": int(cluster_id),
            "dominant_outcome": dominant,
        }
        for label in label_order:
            result[f"{label}_pct"] = _format_percent(float(rates.loc[cluster_id, label]))
        outcome_rows.append(result)
    return cluster_summary, pd.DataFrame(outcome_rows)


def _build_coordinate_frame(reduced: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Prepare the first two PCA dimensions for scatter plotting and report previews."""
    if reduced.shape[1] == 1:
        pc2 = np.zeros(len(reduced), dtype=np.float64)
    else:
        pc2 = reduced[:, 1]
    return pd.DataFrame(
        {
            "pc1": reduced[:, 0],
            "pc2": pc2,
            "cluster": labels.astype(int),
        }
    )


def _ordered_animal_types(raw_features: pd.DataFrame) -> list[str]:
    """Return AnimalType values in Cat/Dog order first, with any unexpected values last."""
    if "AnimalType" not in raw_features.columns:
        raise ValueError("Species-specific clustering requires the raw 'AnimalType' column.")

    values = {
        str(value)
        for value in raw_features["AnimalType"].dropna().unique().tolist()
        if str(value).strip()
    }
    ordered = [species for species in DEFAULT_SPECIES_ORDER if species in values]
    ordered.extend(sorted(values.difference(ordered)))
    return ordered


def run_clustering_analysis(
    raw_features: pd.DataFrame,
    y: Optional[pd.Series] = None,
    k_range: Iterable[int] = DEFAULT_K_RANGE,
    pca_variance: float = DEFAULT_PCA_VARIANCE,
    silhouette_sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    segment_name: str = "All animals",
    include_animal_type: bool = True,
) -> ClusteringAnalysisResult:
    """Run clustering preprocessing, PCA feature reduction, K selection, and K-Means."""
    preprocessing_pipeline = build_clustering_pipeline(include_animal_type=include_animal_type)
    preprocessed = _as_dense_array(preprocessing_pipeline.fit_transform(raw_features))

    # PCA keeps enough components to explain the target variance while reducing one-hot noise.
    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=random_state)
    reduced = pca.fit_transform(preprocessed)

    candidate_scores = _score_kmeans_candidates(
        reduced=reduced,
        k_range=tuple(k_range),
        silhouette_sample_size=silhouette_sample_size,
        random_state=random_state,
    )
    selected_k = _choose_best_k(candidate_scores)

    # Refit the final model at the selected K so all downstream outputs share one label set.
    final_model = KMeans(n_clusters=selected_k, random_state=random_state, n_init=20)
    labels = final_model.fit_predict(reduced)
    selected_score = candidate_scores.loc[
        candidate_scores["k"] == selected_k,
        "silhouette_sample",
    ].iloc[0]

    cluster_summary, outcome_distribution = _summarize_clusters(raw_features, labels, y)
    coordinates = _build_coordinate_frame(reduced, labels)
    centers = pd.DataFrame(
        {
            "cluster": np.arange(selected_k, dtype=int),
            "pc1": final_model.cluster_centers_[:, 0],
            "pc2": (
                final_model.cluster_centers_[:, 1]
                if reduced.shape[1] > 1
                else np.zeros(selected_k, dtype=np.float64)
            ),
        }
    )
    top_breeds = sorted(preprocessing_pipeline.named_steps["breed_top_k"].top_breeds_)

    return ClusteringAnalysisResult(
        segment_name=segment_name,
        raw_shape=raw_features.shape,
        preprocessed_shape=preprocessed.shape,
        pca_shape=reduced.shape,
        pca_variance_threshold=pca_variance,
        pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
        pca_first_two_variance=float(pca.explained_variance_ratio_[:2].sum()),
        selected_k=selected_k,
        inertia=float(final_model.inertia_),
        silhouette_sample=float(selected_score),
        top_breeds=top_breeds,
        candidate_scores=candidate_scores,
        cluster_summary=cluster_summary,
        outcome_distribution=outcome_distribution,
        pca_coordinates=coordinates,
        cluster_centers_2d=centers,
    )


def run_species_clustering_analysis(
    raw_features: pd.DataFrame,
    y: Optional[pd.Series] = None,
    k_range: Iterable[int] = DEFAULT_K_RANGE,
    pca_variance: float = DEFAULT_PCA_VARIANCE,
    silhouette_sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, ClusteringAnalysisResult]:
    """Run the full clustering workflow separately for each AnimalType subset."""
    results: dict[str, ClusteringAnalysisResult] = {}
    for species in _ordered_animal_types(raw_features):
        mask = raw_features["AnimalType"].astype(str) == species
        subset = raw_features.loc[mask].copy()
        y_subset = y.loc[mask].reset_index(drop=True) if y is not None else None

        # AnimalType is the split criterion, so it is intentionally excluded inside each run.
        results[species] = run_clustering_analysis(
            raw_features=subset,
            y=y_subset,
            k_range=k_range,
            pca_variance=pca_variance,
            silhouette_sample_size=silhouette_sample_size,
            random_state=random_state,
            segment_name=species,
            include_animal_type=False,
        )
    return results


def plot_cluster_scatter(
    result: ClusteringAnalysisResult,
    output_path: Path,
) -> Path:
    """Save a PC1/PC2 scatter plot colored by K-Means cluster."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep matplotlib's font/cache files outside the repository and away from unwritable home dirs.
    mpl_config_dir = Path(gettempdir()) / "ds2026-shelter-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate clustering plots.") from exc

    coordinates = result.pca_coordinates
    centers = result.cluster_centers_2d

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    scatter = ax.scatter(
        coordinates["pc1"],
        coordinates["pc2"],
        c=coordinates["cluster"],
        cmap="tab10",
        s=9,
        alpha=0.55,
        linewidths=0,
    )
    ax.scatter(
        centers["pc1"],
        centers["pc2"],
        c=centers["cluster"],
        cmap="tab10",
        s=130,
        marker="X",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title(f"{result.segment_name} K-Means Clusters Projected onto PCA Components")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_species_cluster_scatters(
    results: dict[str, ClusteringAnalysisResult],
    output_path: Path,
) -> Path:
    """Save one PC1/PC2 scatter plot per AnimalType in a single comparison image."""
    if not results:
        raise ValueError("No clustering results were provided for plotting.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep matplotlib's font/cache files outside the repository and away from unwritable home dirs.
    mpl_config_dir = Path(gettempdir()) / "ds2026-shelter-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate clustering plots.") from exc

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(8 * len(results), 6),
        dpi=150,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, result in zip(axes_flat, results.values()):
        coordinates = result.pca_coordinates
        centers = result.cluster_centers_2d
        scatter = ax.scatter(
            coordinates["pc1"],
            coordinates["pc2"],
            c=coordinates["cluster"],
            cmap="tab10",
            s=9,
            alpha=0.55,
            linewidths=0,
        )
        ax.scatter(
            centers["pc1"],
            centers["pc2"],
            c=centers["cluster"],
            cmap="tab10",
            s=130,
            marker="X",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_title(f"{result.segment_name} clusters (K={result.selected_k})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        fig.colorbar(scatter, ax=ax, label="Cluster")

    fig.suptitle("Species-Specific K-Means Clusters Projected onto PCA Components")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavoured markdown table."""
    if df.empty:
        return "_No data available._"
    headers = [str(col) for col in df.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for column, value in row.items():
            if isinstance(value, float):
                if column in {"k", "cluster", "count"} and value.is_integer():
                    cells.append(str(int(value)))
                elif column == "median_age_days":
                    cells.append(f"{value:.1f}")
                elif column == "median_age_years":
                    cells.append(f"{value:.2f}")
                else:
                    cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _append_result_sections(lines: list[str], result: ClusteringAnalysisResult) -> None:
    """Append shape, K selection, profile, and outcome tables for one clustering run."""
    lines.append(f"### {result.segment_name}")
    lines.append("")
    lines.append(f"- Raw feature matrix: {result.raw_shape}")
    lines.append(f"- Encoded clustering matrix: {result.preprocessed_shape}")
    lines.append(f"- PCA-reduced matrix: {result.pca_shape}")
    lines.append(
        f"- PCA explained variance retained: {result.pca_explained_variance:.4f} "
        f"(threshold={result.pca_variance_threshold:.2f})"
    )
    lines.append(f"- PC1 + PC2 explained variance: {result.pca_first_two_variance:.4f}")
    lines.append(f"- Selected K: **{result.selected_k}**")
    lines.append(f"- Final inertia: **{result.inertia:.4f}**")
    lines.append(f"- Sampled silhouette score: **{result.silhouette_sample:.4f}**")
    lines.append("")
    lines.append("#### K Selection")
    lines.append("")
    lines.append(_dataframe_to_markdown(result.candidate_scores))
    lines.append("")
    lines.append("#### Cluster Profiles")
    lines.append("")
    lines.append(_dataframe_to_markdown(result.cluster_summary))
    lines.append("")
    if not result.outcome_distribution.empty:
        lines.append("#### Post-Hoc Outcome Distribution")
        lines.append("")
        lines.append(_dataframe_to_markdown(result.outcome_distribution))
        lines.append("")


def write_clustering_report(
    result: ClusteringAnalysisResult,
    report_path: Path,
    plot_path: Optional[Path] = None,
    plot_error: Optional[str] = None,
) -> Path:
    """Write the clustering process, PCA reduction, metrics, and interpretation to markdown."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Clustering Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report runs only the unsupervised clustering workflow: preprocessing, "
        "PCA feature reduction, K-Means analysis, and cluster interpretation."
    )
    lines.append("")
    lines.append("`OutcomeType` is not used to fit clusters; it is shown only after clustering for interpretation.")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append("- Raw input: shelter animal feature columns only")
    lines.append("- Feature engineering: age, date/time, sex/neuter status, breed, color, and name signals")
    lines.append("- Breed reduction: keep top 30 `primary_breed` values and group the rest as `Other`")
    lines.append("- Numeric preprocessing: median imputation and RobustScaler")
    lines.append("- Categorical preprocessing: most-frequent imputation and OneHotEncoder")
    lines.append("- PCA: retain at least 95% of variance before K-Means")
    lines.append("- Model: K-Means, selected by sampled silhouette score over K=2..6")
    lines.append("")
    lines.append("## Shape And Reduction")
    lines.append("")
    lines.append(f"- Raw feature matrix: {result.raw_shape}")
    lines.append(f"- Encoded clustering matrix: {result.preprocessed_shape}")
    lines.append(f"- PCA-reduced matrix: {result.pca_shape}")
    lines.append(
        f"- PCA explained variance retained: {result.pca_explained_variance:.4f} "
        f"(threshold={result.pca_variance_threshold:.2f})"
    )
    lines.append(f"- PC1 + PC2 explained variance: {result.pca_first_two_variance:.4f}")
    lines.append("")
    lines.append("## K Selection")
    lines.append("")
    lines.append(_dataframe_to_markdown(result.candidate_scores))
    lines.append("")
    lines.append(f"Selected K: **{result.selected_k}**")
    lines.append(f"Final inertia: **{result.inertia:.4f}**")
    lines.append(f"Sampled silhouette score: **{result.silhouette_sample:.4f}**")
    lines.append("")
    lines.append("## Cluster Profiles")
    lines.append("")
    lines.append(_dataframe_to_markdown(result.cluster_summary))
    lines.append("")
    if not result.outcome_distribution.empty:
        lines.append("## Post-Hoc Outcome Distribution")
        lines.append("")
        lines.append(_dataframe_to_markdown(result.outcome_distribution))
        lines.append("")
    lines.append("## Visualization")
    lines.append("")
    if plot_path is not None:
        lines.append(f"- PCA scatter plot: `{plot_path.as_posix()}`")
        lines.append("- Each point is one animal projected onto PC1 and PC2.")
        lines.append("- Colors are K-Means clusters; X markers are cluster centers in PCA space.")
    elif plot_error:
        lines.append(f"- Plot was skipped: {plot_error}")
    else:
        lines.append("- Plot was not requested.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "The PCA scatter is a 2D explanation view, while K-Means is fit on the PCA matrix "
        "that retains the configured variance threshold."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_species_clustering_report(
    results: dict[str, ClusteringAnalysisResult],
    report_path: Path,
    plot_path: Optional[Path] = None,
    plot_error: Optional[str] = None,
) -> Path:
    """Write a Cat/Dog-separated clustering report with PCA metrics and interpretation."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Species-Specific Clustering Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report runs the unsupervised clustering workflow separately for each "
        "`AnimalType` subset. Cats and dogs are filtered first because species can "
        "dominate distance-based clustering when they are mixed in one K-Means run."
    )
    lines.append("")
    lines.append("`OutcomeType` is not used to fit clusters; it is shown only after clustering for interpretation.")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append("- Split raw data by `AnimalType` (`Cat`, `Dog`)")
    lines.append("- Run feature engineering independently inside each species subset")
    lines.append("- Exclude `animal_type` from clustering features because it is constant after filtering")
    lines.append("- Breed reduction: keep top 30 `primary_breed` values per species and group the rest as `Other`")
    lines.append("- Numeric preprocessing: median imputation and RobustScaler")
    lines.append("- Categorical preprocessing: most-frequent imputation and OneHotEncoder")
    lines.append("- PCA: retain at least 95% of variance before K-Means")
    lines.append("- Model: K-Means, selected by sampled silhouette score over K=2..6 per species")
    lines.append("")
    lines.append("## Species Results")
    lines.append("")
    for result in results.values():
        _append_result_sections(lines, result)

    lines.append("## Visualization")
    lines.append("")
    if plot_path is not None:
        lines.append(f"- PCA scatter plot: `{plot_path.as_posix()}`")
        lines.append("- Each panel is one species-specific clustering run.")
        lines.append("- Each point is one animal projected onto PC1 and PC2.")
        lines.append("- Colors are K-Means clusters; X markers are cluster centers in PCA space.")
    elif plot_error:
        lines.append(f"- Plot was skipped: {plot_error}")
    else:
        lines.append("- Plot was not requested.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "The PCA scatter uses only PC1 and PC2 for visualization. K-Means is fit on "
        "the PCA matrix that retains the configured variance threshold for each species."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
