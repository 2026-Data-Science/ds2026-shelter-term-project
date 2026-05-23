from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.clustering.preprocessing import (
    BREED_MIN_SPECIES_RATIO,
    CATEGORICAL_FEATURES,
    ENRICHED_TRAIN_PATH,
    LOS_LABELS,
    NUMERIC_FEATURES,
    PROFILE_ONLY_FEATURES,
    PROJECT_ROOT,
    TRAIN_CSV,
    build_long_stay_clustering_frame,
    build_long_stay_preprocessor,
    build_train_with_latest_intake,
    find_intake_csv,
    validate_columns,
)


if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "long_stay_risk_clustering"
REPORT_PATH = OUTPUT_DIR / "long_stay_risk_clustering_report.md"
PCA_SCATTER_FILENAME = "long_stay_cluster_pca_scatter.png"
DURATION_BOXPLOT_FILENAME = "duration_by_cluster_boxplot.png"
DURATION_BINS_FILENAME = "duration_bins_by_cluster.png"

DEFAULT_K_RANGE = (2, 3, 4, 5, 6)
DEFAULT_RANDOM_STATE = 42
DEFAULT_PCA_VARIANCE = 0.95
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 3000


@dataclass
class SpeciesLongStayClusteringResult:
    """Container for one species-specific long-stay risk clustering run."""

    species: str
    raw_shape: tuple[int, int]
    feature_shape: tuple[int, int]
    encoded_shape: tuple[int, int]
    pca_shape: tuple[int, int]
    pca_explained_variance: float
    selected_k: int
    silhouette_sample: float
    candidate_scores: pd.DataFrame
    cluster_profile: pd.DataFrame
    duration_bin_profile: pd.DataFrame
    outcome_profile: pd.DataFrame
    feature_separation: pd.DataFrame
    pca_coordinates: pd.DataFrame
    cluster_centers_2d: pd.DataFrame


def _top_value(series: pd.Series) -> str:
    """Return the most common value with its within-cluster percentage."""
    cleaned = series.fillna("Unknown").astype(str)
    if cleaned.empty:
        return "Unknown (0.0%)"
    counts = cleaned.value_counts(dropna=False)
    value = counts.index[0]
    ratio = counts.iloc[0] / len(cleaned) * 100.0
    return f"{value} ({ratio:.1f}%)"


def _format_percent(value: float) -> str:
    """Format a ratio already expressed as 0-100 as a percentage string."""
    return f"{value:.2f}%"


def _as_dense(matrix: object) -> np.ndarray:
    """Convert sklearn matrix outputs to dense float arrays for PCA/K-Means."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _markdown_value(value: object) -> str:
    """Format values for compact markdown tables without optional dependencies."""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if frame.empty:
        return "_No data available._"
    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_markdown_value(value) for value in row) + " |")
    return "\n".join(rows)


def _select_pca_components(explained: np.ndarray) -> int:
    """Choose the smallest PCA component count that reaches the variance threshold."""
    cumulative = np.cumsum(explained)
    return int(np.searchsorted(cumulative, DEFAULT_PCA_VARIANCE, side="left") + 1)


def _score_kmeans(reduced: np.ndarray) -> pd.DataFrame:
    """Grid-search K=2..6 with sampled silhouette score."""
    rows: list[dict[str, float]] = []
    sample_size: Optional[int] = (
        DEFAULT_SILHOUETTE_SAMPLE_SIZE if len(reduced) > DEFAULT_SILHOUETTE_SAMPLE_SIZE else None
    )
    for k in DEFAULT_K_RANGE:
        model = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE, n_init=20)
        labels = model.fit_predict(reduced)
        score = silhouette_score(
            reduced,
            labels,
            sample_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE,
        )
        rows.append({"k": int(k), "inertia": float(model.inertia_), "silhouette_sample": float(score)})
    return pd.DataFrame(rows)


def _rank_feature_separation(
    encoded: np.ndarray,
    encoded_feature_names: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Rank input columns by how strongly their encoded values vary across clusters."""
    global_mean = encoded.mean(axis=0)
    total_ss = ((encoded - global_mean) ** 2).sum(axis=0)
    between_ss = np.zeros(encoded.shape[1], dtype=float)
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        cluster_mean = encoded[mask].mean(axis=0)
        between_ss += mask.sum() * ((cluster_mean - global_mean) ** 2)

    encoded_scores = np.divide(
        between_ss,
        total_ss,
        out=np.zeros_like(between_ss),
        where=total_ss > 0,
    )
    rows: list[dict[str, object]] = []
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if feature in NUMERIC_FEATURES:
            indexes = [index for index, name in enumerate(encoded_feature_names) if name == feature]
        else:
            prefix = f"{feature}_"
            indexes = [
                index for index, name in enumerate(encoded_feature_names) if name.startswith(prefix)
            ]
        rows.append(
            {
                "feature": feature,
                "separation_score": round(float(encoded_scores[indexes].sum()), 4) if indexes else 0.0,
                "avg_encoded_eta2": round(float(encoded_scores[indexes].mean()), 4) if indexes else 0.0,
                "encoded_columns": int(len(indexes)),
            }
        )
    return pd.DataFrame(rows).sort_values("separation_score", ascending=False).reset_index(drop=True)


def _profile_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Profile clusters with post-hoc breed/color/mix, duration, and outcome distributions."""
    rows: list[dict[str, object]] = []
    total = len(frame)
    for cluster_id in sorted(frame["cluster"].unique()):
        subset = frame[frame["cluster"] == cluster_id]
        duration = subset["length_of_stay_days"]
        rows.append(
            {
                "cluster": int(cluster_id),
                "count": int(len(subset)),
                "ratio": _format_percent(len(subset) / total * 100.0),
                "top_intake_type": _top_value(subset["intake_type"]),
                "top_condition": _top_value(subset["intake_condition"]),
                "top_age_group": _top_value(subset["intake_age_group"]),
                "top_sex": _top_value(subset["sex"]),
                "top_neuter_status": _top_value(subset["neuter_status"]),
                "has_name": _format_percent(float(subset["has_name"].mean()) * 100.0),
                "top_breed": _top_value(subset["primary_breed"]),
                "top_color": _top_value(subset["primary_color"]),
                "mixed_breed": _format_percent(float(subset["is_mixed_breed"].mean()) * 100.0),
                "duration_mean_days": round(float(duration.mean()), 2),
                "duration_median_days": round(float(duration.median()), 2),
                "duration_p75_days": round(float(duration.quantile(0.75)), 2),
                "duration_p90_days": round(float(duration.quantile(0.90)), 2),
                "long_stay_30": _format_percent(float(subset["long_stay_30"].mean()) * 100.0),
                "long_stay_60": _format_percent(float(subset["long_stay_60"].mean()) * 100.0),
                "long_stay_90": _format_percent(float(subset["long_stay_90"].mean()) * 100.0),
                "top_duration_category": _top_value(subset["duration_category"]),
            }
        )
    profile = pd.DataFrame(rows)

    bin_profile = pd.crosstab(
        frame["cluster"],
        frame["duration_category"],
        normalize="index",
    ).mul(100.0)
    bin_profile = bin_profile.reindex(columns=LOS_LABELS, fill_value=0.0).reset_index()

    outcome_profile = pd.crosstab(
        frame["cluster"],
        frame["OutcomeType"],
        normalize="index",
    ).mul(100.0).reset_index()
    return profile, bin_profile, outcome_profile


def run_species_long_stay_clustering(
    species_frame: pd.DataFrame,
    species: str,
) -> SpeciesLongStayClusteringResult:
    """Run PCA + K-Means on duration-free long-stay clustering inputs."""
    frame = species_frame.copy()
    preprocessor = build_long_stay_preprocessor()
    encoded = _as_dense(preprocessor.fit_transform(frame))

    pca = PCA(svd_solver="full", random_state=DEFAULT_RANDOM_STATE)
    full_reduced = pca.fit_transform(encoded)
    component_count = _select_pca_components(pca.explained_variance_ratio_)
    reduced = full_reduced[:, :component_count]

    candidate_scores = _score_kmeans(reduced)
    selected_row = candidate_scores.sort_values(
        by=["silhouette_sample", "k"],
        ascending=[False, True],
    ).iloc[0]
    selected_k = int(selected_row["k"])

    final_model = KMeans(n_clusters=selected_k, random_state=DEFAULT_RANDOM_STATE, n_init=20)
    frame["cluster"] = final_model.fit_predict(reduced)

    cluster_profile, duration_bin_profile, outcome_profile = _profile_clusters(frame)
    feature_separation = _rank_feature_separation(
        encoded=encoded,
        encoded_feature_names=list(preprocessor.get_feature_names_out()),
        labels=frame["cluster"].to_numpy(dtype=int),
    )
    pca_coordinates = pd.DataFrame(
        {
            "pc1": reduced[:, 0],
            "pc2": reduced[:, 1] if reduced.shape[1] > 1 else np.zeros(len(reduced)),
            "cluster": frame["cluster"].to_numpy(dtype=int),
        }
    )
    cluster_centers = pd.DataFrame(
        {
            "cluster": np.arange(selected_k, dtype=int),
            "pc1": final_model.cluster_centers_[:, 0],
            "pc2": (
                final_model.cluster_centers_[:, 1]
                if reduced.shape[1] > 1
                else np.zeros(selected_k)
            ),
        }
    )

    return SpeciesLongStayClusteringResult(
        species=species,
        raw_shape=species_frame.shape,
        feature_shape=frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].shape,
        encoded_shape=encoded.shape,
        pca_shape=reduced.shape,
        pca_explained_variance=float(pca.explained_variance_ratio_[:component_count].sum()),
        selected_k=selected_k,
        silhouette_sample=float(selected_row["silhouette_sample"]),
        candidate_scores=candidate_scores,
        cluster_profile=cluster_profile,
        duration_bin_profile=duration_bin_profile,
        outcome_profile=outcome_profile,
        feature_separation=feature_separation,
        pca_coordinates=pca_coordinates,
        cluster_centers_2d=cluster_centers,
    )


def run_long_stay_clustering(frame: pd.DataFrame) -> tuple[dict[str, SpeciesLongStayClusteringResult], pd.DataFrame]:
    """Run long-stay clustering separately for Cat and Dog subsets."""
    results: dict[str, SpeciesLongStayClusteringResult] = {}
    clustered_frames: list[pd.DataFrame] = []
    for species in ["Cat", "Dog"]:
        subset = frame[frame["AnimalType"].astype(str) == species].copy()
        if subset.empty:
            continue
        result = run_species_long_stay_clustering(subset, species)
        results[species] = result
        subset = subset.reset_index(drop=True)
        subset["cluster"] = result.pca_coordinates["cluster"].to_numpy(dtype=int)
        clustered_frames.append(subset)
    clustered = pd.concat(clustered_frames, ignore_index=True)
    return results, clustered


def _configure_matplotlib() -> None:
    """Keep matplotlib cache files outside the repository and unwritable home directories."""
    mpl_config_dir = Path(gettempdir()) / "ds2026-shelter-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def plot_pca_scatter(results: dict[str, SpeciesLongStayClusteringResult], output_path: Path) -> Path:
    """Save species-specific PCA scatter plots with cluster centers."""
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6), dpi=150, squeeze=False)
    for ax, result in zip(axes.ravel(), results.values()):
        coords = result.pca_coordinates
        centers = result.cluster_centers_2d
        scatter = ax.scatter(
            coords["pc1"],
            coords["pc2"],
            c=coords["cluster"],
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
        ax.set_title(f"{result.species} clusters (K={result.selected_k})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.suptitle("Long-Stay Risk Clusters Projected onto PCA Components")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_duration_boxplots(clustering_frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot numerical duration distributions by species and cluster."""
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    species_values = sorted(clustering_frame["AnimalType"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(species_values), figsize=(8 * len(species_values), 6), dpi=150, squeeze=False)
    for ax, species in zip(axes.ravel(), species_values):
        subset = clustering_frame[clustering_frame["AnimalType"] == species]
        clusters = sorted(subset["cluster"].unique())
        values = [
            subset.loc[subset["cluster"] == cluster, "length_of_stay_days"].dropna() + 0.1
            for cluster in clusters
        ]
        ax.boxplot(values, showfliers=False, patch_artist=True)
        ax.set_xticks(range(1, len(clusters) + 1))
        ax.set_xticklabels([str(cluster) for cluster in clusters])
        ax.set_yscale("log")
        ax.set_title(f"{species} duration by cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Length of stay days (+0.1, log scale)")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_duration_bins(clustering_frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot categorical duration-bin distribution by species and cluster."""
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    species_values = sorted(clustering_frame["AnimalType"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(species_values), figsize=(8 * len(species_values), 6), dpi=150, squeeze=False)
    for ax, species in zip(axes.ravel(), species_values):
        subset = clustering_frame[clustering_frame["AnimalType"] == species]
        table = pd.crosstab(subset["cluster"], subset["duration_category"], normalize="index").mul(100.0)
        table = table.reindex(columns=LOS_LABELS, fill_value=0.0)
        table.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_title(f"{species} duration categories by cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Percentage within cluster")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Duration", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def write_long_stay_report(
    results: dict[str, SpeciesLongStayClusteringResult],
    enriched: pd.DataFrame,
    clustering_frame: pd.DataFrame,
    plot_paths: dict[str, Path],
    report_path: Path = REPORT_PATH,
) -> Path:
    """Write long-stay clustering metrics and duration profiles to markdown."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Long-Stay Risk Clustering Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This workflow joins each outcome row to the nearest previous intake row, "
        "computes length of stay, clusters animals using duration-free intake/animal "
        "features, and profiles each cluster with breed/sex/color/mixed-breed, numerical duration, "
        "categorical duration, and outcome distributions."
    )
    lines.append("")
    lines.append("## Workflow Design")
    lines.append("")
    lines.append("Main long-stay clustering uses only these inputs:")
    lines.append("")
    lines.append("- `has_name`")
    lines.append("- `primary_breed`")
    lines.append("- `intake_type`")
    lines.append("- `intake_condition`")
    lines.append("- `neuter_status`")
    lines.append("- `intake_age_group`")
    lines.append("")
    lines.append(
        "Breed is represented as `primary_breed`; species-specific breed values under "
        f"{BREED_MIN_SPECIES_RATIO:.0%} are grouped into `Other` before clustering."
    )
    lines.append("")
    lines.append("Post-hoc-only profiling uses these fields after clusters are assigned:")
    lines.append("")
    lines.append("- Sex: `sex`")
    lines.append("- Color: `primary_color`")
    lines.append("- Mixed breed flag: `is_mixed_breed`")
    lines.append("- Numerical duration: `length_of_stay_days`")
    lines.append("- Categorical duration: `duration_category`")
    lines.append("- Outcome distribution: `OutcomeType`")
    lines.append("")
    lines.append("## Data Join And Duration")
    lines.append("")
    lines.append(f"- Enriched CSV: `{ENRICHED_TRAIN_PATH.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Original train rows: {len(enriched):,}")
    lines.append(f"- Rows usable for clustering/profile: {len(clustering_frame):,}")
    lines.append(f"- Missing or invalid duration rows excluded from clustering/profile: {len(enriched) - len(clustering_frame):,}")
    lines.append("")
    lines.append("Matching rule: same animal ID, then latest intake datetime at or before outcome datetime.")
    lines.append("")
    lines.append("Duration is calculated as `outcome DateTime - intake_datetime`.")
    lines.append("")
    lines.append("## Missing Handling")
    lines.append("")
    lines.append("- Original rows with no valid previous intake or invalid duration are kept in the enriched CSV.")
    lines.append("- Rows without valid non-negative duration are excluded from clustering/profile tables.")
    lines.append("- Categorical clustering inputs are imputed with the most frequent value after Unknown normalization.")
    lines.append("- Numeric clustering inputs are imputed with the median and scaled with RobustScaler.")
    lines.append("")
    lines.append("## Clustering Inputs")
    lines.append("")
    lines.append(
        "Duration, sex, color, and mixed-breed columns are not used as clustering inputs. "
        "`primary_breed` is used as a categorical clustering input."
    )
    lines.append("")
    lines.append("Numeric inputs:")
    for column in NUMERIC_FEATURES:
        lines.append(f"- `{column}`")
    lines.append("")
    lines.append("Categorical inputs:")
    for column in CATEGORICAL_FEATURES:
        lines.append(f"- `{column}`")
    lines.append("")
    lines.append("## Post-Hoc Profiling Fields")
    lines.append("")
    for column in PROFILE_ONLY_FEATURES:
        lines.append(f"- `{column}`")
    lines.append("- `OutcomeType`")
    lines.append("")
    lines.append("## Species Results")
    lines.append("")
    for result in results.values():
        lines.append(f"### {result.species}")
        lines.append("")
        lines.append(f"- Raw rows/columns: {result.raw_shape}")
        lines.append(f"- Feature matrix: {result.feature_shape}")
        lines.append(f"- Encoded matrix: {result.encoded_shape}")
        lines.append(f"- PCA matrix: {result.pca_shape}")
        lines.append(f"- PCA explained variance retained: {result.pca_explained_variance:.4f}")
        lines.append(f"- Selected K: **{result.selected_k}**")
        lines.append(f"- Sampled silhouette score: **{result.silhouette_sample:.4f}**")
        lines.append("")
        lines.append("#### K Selection")
        lines.append("")
        lines.append(_markdown_table(result.candidate_scores))
        lines.append("")
        lines.append("#### Feature Separation Ranking")
        lines.append("")
        lines.append(
            "This is not supervised feature importance; it ranks clustering inputs by "
            "how differently they are distributed across the selected clusters."
        )
        lines.append("")
        lines.append(_markdown_table(result.feature_separation))
        lines.append("")
        lines.append("#### Cluster Profiles With Numerical Duration")
        lines.append("")
        lines.append(_markdown_table(result.cluster_profile))
        lines.append("")
        lines.append("#### Categorical Duration Distribution By Cluster")
        lines.append("")
        lines.append(_markdown_table(result.duration_bin_profile))
        lines.append("")
        lines.append("#### Post-Hoc Outcome Distribution By Cluster")
        lines.append("")
        lines.append(_markdown_table(result.outcome_profile))
        lines.append("")
    lines.append("## Visualizations")
    lines.append("")
    for label, path in plot_paths.items():
        lines.append(f"- {label}: `{path.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_long_stay_workflow(
    train_csv: Path = TRAIN_CSV,
    intake_csv: Optional[Path] = None,
    enriched_train_path: Path = ENRICHED_TRAIN_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[dict[str, SpeciesLongStayClusteringResult], pd.DataFrame, pd.DataFrame]:
    """Run the full long-stay workflow and write data, reports, and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_train_path.parent.mkdir(parents=True, exist_ok=True)

    intake_csv = intake_csv or find_intake_csv()
    train = pd.read_csv(train_csv)
    intakes = pd.read_csv(intake_csv)
    validate_columns(train, ["AnimalID", "DateTime", "OutcomeType", "AnimalType"], train_csv.name)
    validate_columns(
        intakes,
        [
            "animal_id",
            "datetime",
            "intake_type",
            "intake_condition",
            "sex_upon_intake",
            "age_upon_intake",
            "breed",
            "color",
        ],
        intake_csv.name,
    )

    enriched = build_train_with_latest_intake(train, intakes)
    enriched.to_csv(enriched_train_path, index=False)

    clustering_frame = build_long_stay_clustering_frame(enriched)
    results, clustered = run_long_stay_clustering(clustering_frame)
    clustered.to_csv(output_dir / "clustered_long_stay_profile_rows.csv", index=False)

    plot_paths = {
        "PCA cluster scatter": output_dir / PCA_SCATTER_FILENAME,
        "Numerical duration boxplots": output_dir / DURATION_BOXPLOT_FILENAME,
        "Categorical duration bins": output_dir / DURATION_BINS_FILENAME,
    }
    plot_pca_scatter(results, plot_paths["PCA cluster scatter"])
    plot_duration_boxplots(clustered, plot_paths["Numerical duration boxplots"])
    plot_duration_bins(clustered, plot_paths["Categorical duration bins"])
    write_long_stay_report(results, enriched, clustering_frame, plot_paths, output_dir / REPORT_PATH.name)
    return results, enriched, clustered


def main() -> None:
    """CLI entrypoint for the long-stay risk clustering workflow."""
    results, enriched, clustered = run_long_stay_workflow()
    print(f"Enriched CSV written: {ENRICHED_TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows usable for clustering/profile: {len(clustered):,} / {len(enriched):,}")
    for species, result in results.items():
        print(
            f"{species}: K={result.selected_k}, "
            f"silhouette={result.silhouette_sample:.4f}, "
            f"encoded={result.encoded_shape}, pca={result.pca_shape}"
        )
    print(f"Report written: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Plot written: {(OUTPUT_DIR / PCA_SCATTER_FILENAME).relative_to(PROJECT_ROOT)}")
    print(f"Plot written: {(OUTPUT_DIR / DURATION_BOXPLOT_FILENAME).relative_to(PROJECT_ROOT)}")
    print(f"Plot written: {(OUTPUT_DIR / DURATION_BINS_FILENAME).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
