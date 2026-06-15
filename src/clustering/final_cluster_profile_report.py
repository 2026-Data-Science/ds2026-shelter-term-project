"""Section 5.5 report generator: profiles the final production clustering labels.

Takes the already-fitted cluster labels and builds the report tables and figures:
per-cluster profiles (top intake/breed/age, duration stats, long-stay ratios,
outcome mix), long-stay candidate clusters per species, and the silhouette
re-check in both PCA and full space. Re-emphasises that duration/outcome were
used only for profiling, never for K-Means fitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.clustering.analysis import (
    DEFAULT_PCA_VARIANCE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SpeciesLongStayClusteringResult,
    _as_dense,
    _markdown_table,
    _select_pca_components,
    _top_value,
    plot_duration_bins,
    plot_duration_boxplots,
    plot_pca_scatter,
)
from src.clustering.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_long_stay_preprocessor,
)

PROFILE_CSV = OUTPUT_DIR / "final_cluster_profile_for_report.csv"
PROFILE_MD = OUTPUT_DIR / "final_cluster_profile_for_report.md"

OUTCOME_LABELS = [
    "Adoption",
    "Died",
    "Euthanasia",
    "Return_to_owner",
    "Transfer",
]


def _silhouette_sample_size(n_samples: int) -> int | None:
    if n_samples > DEFAULT_SILHOUETTE_SAMPLE_SIZE:
        return DEFAULT_SILHOUETTE_SAMPLE_SIZE
    return None


def _outcome_ratio(subset: pd.DataFrame, label: str) -> float:
    return round(float((subset["OutcomeType"].astype(str) == label).mean()) * 100.0, 2)


def _collect_production_settings(
    species_frame: pd.DataFrame,
    result: SpeciesLongStayClusteringResult,
) -> dict[str, object]:
    # Re-encode and re-run PCA to recompute silhouette in both spaces on the same labels.
    preprocessor = build_long_stay_preprocessor()
    encoded = _as_dense(
        preprocessor.fit_transform(species_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    )
    pca = PCA(n_components=encoded.shape[1], svd_solver="full", random_state=DEFAULT_RANDOM_STATE)
    full_reduced = pca.fit_transform(encoded)
    n_components = _select_pca_components(pca.explained_variance_ratio_)
    reduced = full_reduced[:, :n_components]
    labels = species_frame["cluster"].to_numpy(dtype=int)
    sample_size = _silhouette_sample_size(len(reduced))
    # PCA-space silhouette: how separated the clusters look in the reduced embedding.
    sil_pca = float(
        silhouette_score(
            reduced,
            labels,
            sample_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE,
        )
    )
    # Full-space silhouette: the same labels scored against all encoded features.
    sil_full = float(
        silhouette_score(
            encoded,
            labels,
            sample_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE,
        )
    )
    return {
        "species": result.species,
        "pca_threshold": DEFAULT_PCA_VARIANCE,
        "n_components_selected": int(n_components),
        "explained_variance_ratio_sum": round(
            float(pca.explained_variance_ratio_[:n_components].sum()),
            6,
        ),
        "encoded_dimensions": int(encoded.shape[1]),
        "k": int(result.selected_k),
        "silhouette_pca_space": round(sil_pca, 6),
        "silhouette_full_space": round(sil_full, 6),
        "silhouette_gap_pca_minus_full": round(sil_pca - sil_full, 6),
    }


def _build_cluster_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for species in sorted(clustered["AnimalType"].astype(str).unique()):
        species_frame = clustered[clustered["AnimalType"].astype(str) == species]
        total = len(species_frame)
        for cluster_id in sorted(species_frame["cluster"].unique()):
            subset = species_frame[species_frame["cluster"] == cluster_id]
            duration = subset["length_of_stay_days"]
            rows.append(
                {
                    "species": species,
                    "cluster": int(cluster_id),
                    "count": int(len(subset)),
                    "ratio_pct": round(len(subset) / total * 100.0, 2),
                    "top_intake_type": _top_value(subset["intake_type"]),
                    "top_intake_condition": _top_value(subset["intake_condition"]),
                    "top_neuter_status": _top_value(subset["neuter_status"]),
                    "top_intake_age_group": _top_value(subset["intake_age_group"]),
                    "top_primary_breed": _top_value(subset["primary_breed"]),
                    "has_name_ratio_pct": round(float(subset["has_name"].mean()) * 100.0, 2),
                    "duration_mean_days": round(float(duration.mean()), 2),
                    "duration_median_days": round(float(duration.median()), 2),
                    "duration_p75_days": round(float(duration.quantile(0.75)), 2),
                    "duration_p90_days": round(float(duration.quantile(0.90)), 2),
                    "long_stay_30_ratio_pct": round(
                        float(subset["long_stay_30"].mean()) * 100.0,
                        2,
                    ),
                    "long_stay_60_ratio_pct": round(
                        float(subset["long_stay_60"].mean()) * 100.0,
                        2,
                    ),
                    "top_outcome_type": _top_value(subset["OutcomeType"]),
                    "adoption_ratio_pct": _outcome_ratio(subset, "Adoption"),
                    "transfer_ratio_pct": _outcome_ratio(subset, "Transfer"),
                    "return_to_owner_ratio_pct": _outcome_ratio(subset, "Return_to_owner"),
                    "euthanasia_ratio_pct": _outcome_ratio(subset, "Euthanasia"),
                    "died_ratio_pct": _outcome_ratio(subset, "Died"),
                }
            )
    return pd.DataFrame(rows)


def _long_stay_candidates(profiles: pd.DataFrame, species: str, top_n: int = 2) -> pd.DataFrame:
    subset = profiles[profiles["species"] == species].copy()
    # Rank clusters by combined long-stay-30 ratio and median duration; lower score = stronger.
    subset["long_stay_rank_score"] = (
        subset["long_stay_30_ratio_pct"].rank(ascending=False, method="dense")
        + subset["duration_median_days"].rank(ascending=False, method="dense")
    )
    return (
        subset.sort_values(
            by=["long_stay_rank_score", "long_stay_30_ratio_pct", "duration_median_days"],
            ascending=[True, False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def write_final_cluster_profile_report(
    settings_df: pd.DataFrame,
    profiles: pd.DataFrame,
    cat_candidates: pd.DataFrame,
    dog_candidates: pd.DataFrame,
    plot_paths: dict[str, Path],
    report_path: Path = PROFILE_MD,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Final Cluster Profile For Report (Section 5.5)")
    lines.append("")
    lines.append("## Final Clustering Setting Used For Profiling")
    lines.append("")
    lines.append(
        "Production long-stay clustering uses PCA variance threshold "
        f"**{DEFAULT_PCA_VARIANCE}**, k selected by highest silhouette in PCA space "
        f"(k in {list(range(2, 7))}), and duration-free intake/animal features only."
    )
    lines.append("")
    lines.append(_markdown_table(settings_df))
    lines.append("")

    for species in ["Cat", "Dog"]:
        lines.append(f"## {species} Cluster Profile Summary")
        lines.append("")
        species_profiles = profiles[profiles["species"] == species]
        lines.append(
            _markdown_table(
                species_profiles[
                    [
                        "cluster",
                        "count",
                        "ratio_pct",
                        "top_intake_type",
                        "top_intake_condition",
                        "top_neuter_status",
                        "top_intake_age_group",
                        "top_primary_breed",
                        "has_name_ratio_pct",
                        "duration_mean_days",
                        "duration_median_days",
                        "long_stay_30_ratio_pct",
                        "long_stay_60_ratio_pct",
                        "top_outcome_type",
                    ]
                ]
            )
        )
        lines.append("")

    lines.append("## Long-Stay Risk Cluster Candidates")
    lines.append("")
    lines.append(
        "The clusters below are **associated with** higher long-stay indicators "
        "(not causal claims)."
    )
    lines.append("")
    lines.append("### Cat (top candidates)")
    lines.append("")
    for _, row in cat_candidates.iterrows():
        lines.append(
            f"- Cluster **{int(row['cluster'])}**: median stay {row['duration_median_days']:.1f} days, "
            f"long_stay_30={row['long_stay_30_ratio_pct']:.1f}%, "
            f"long_stay_60={row['long_stay_60_ratio_pct']:.1f}%, "
            f"top intake type {row['top_intake_type']}, top age group {row['top_intake_age_group']}."
        )
    lines.append("")
    lines.append("### Dog (top candidates)")
    lines.append("")
    for _, row in dog_candidates.iterrows():
        lines.append(
            f"- Cluster **{int(row['cluster'])}**: median stay {row['duration_median_days']:.1f} days, "
            f"long_stay_30={row['long_stay_30_ratio_pct']:.1f}%, "
            f"long_stay_60={row['long_stay_60_ratio_pct']:.1f}%, "
            f"top intake type {row['top_intake_type']}, top age group {row['top_intake_age_group']}."
        )
    lines.append("")

    lines.append("## Interpretation Notes For Figure 4 PCA Scatter")
    lines.append("")
    lines.append(
        f"- Source: `{plot_paths['PCA cluster scatter'].relative_to(PROJECT_ROOT)}`"
    )
    lines.append(
        "- Points are animals projected onto the first two PCA components of the "
        "**production** clustering embedding (PCA threshold 0.95)."
    )
    lines.append(
        "- Colors show final K-Means cluster labels used throughout Section 5.5 profiling."
    )
    lines.append(
        "- This figure is for visual separation inspection; PCA scatter axes are not "
        "directly interpretable as single raw variables."
    )
    lines.append("")

    lines.append("## Interpretation Notes For Figure 5 Duration Boxplot")
    lines.append("")
    lines.append(
        f"- Source: `{plot_paths['Numerical duration boxplots'].relative_to(PROJECT_ROOT)}`"
    )
    lines.append(
        "- Boxplots use post-hoc `length_of_stay_days` by final cluster label (log scale)."
    )
    lines.append(
        "- Duration was **not** used in K-Means fitting; it is shown only to profile "
        "whether intake/animal clusters differ in observed stay length."
    )
    lines.append("")

    lines.append("## Interpretation Notes For Figure 6 Duration Category Distribution")
    lines.append("")
    lines.append(
        f"- Source: `{plot_paths['Categorical duration bins'].relative_to(PROJECT_ROOT)}`"
    )
    lines.append(
        "- Stacked bars show within-cluster shares of categorical duration bins "
        "(`duration_category`), derived post-hoc from outcome minus intake datetime."
    )
    lines.append(
        "- Use this figure to compare long-stay bin concentration across clusters "
        "without treating duration as a clustering input."
    )
    lines.append("")

    lines.append("## Caution: Post-Hoc Profiling Variables")
    lines.append("")
    lines.append("K-Means fitting used only:")
    lines.append("")
    lines.append("- `has_name`")
    for feature in CATEGORICAL_FEATURES:
        lines.append(f"- `{feature}`")
    lines.append("")
    lines.append("Not used in fitting (profiling only):")
    lines.append("")
    lines.append(
        "- `length_of_stay_days`, `length_of_stay_hours`, `duration_category`, "
        "`OutcomeType`, `sex`, `primary_color`, `is_mixed_breed`, and related outcome ratios."
    )
    lines.append("")

    lines.append("## Status")
    lines.append("")
    lines.append("- **5.5 profiling outputs were generated from the final clustering labels.**")
    lines.append("- **Post-hoc variables were not used in K-Means fitting.**")
    lines.append("- **classification pipeline was not modified.**")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_final_cluster_profile_report(
    clustering_frame: pd.DataFrame,
    results: dict[str, SpeciesLongStayClusteringResult],
    clustered: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    plot_paths: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Section 5.5 report tables from production clustering labels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    settings_rows: list[dict[str, object]] = []
    for species, result in results.items():
        species_frame = clustered[clustered["AnimalType"].astype(str) == species].copy()
        settings_rows.append(_collect_production_settings(species_frame, result))
    settings_df = pd.DataFrame(settings_rows)

    profiles = _build_cluster_profiles(clustered)
    cat_candidates = _long_stay_candidates(profiles, "Cat", top_n=2)
    dog_candidates = _long_stay_candidates(profiles, "Dog", top_n=2)

    if plot_paths is None:
        from src.clustering.analysis import (
            DURATION_BINS_FILENAME,
            DURATION_BOXPLOT_FILENAME,
            PCA_SCATTER_FILENAME,
        )

        plot_paths = {
            "PCA cluster scatter": output_dir / PCA_SCATTER_FILENAME,
            "Numerical duration boxplots": output_dir / DURATION_BOXPLOT_FILENAME,
            "Categorical duration bins": output_dir / DURATION_BINS_FILENAME,
        }

    plot_pca_scatter(results, plot_paths["PCA cluster scatter"])
    plot_duration_boxplots(clustered, plot_paths["Numerical duration boxplots"])
    plot_duration_bins(clustered, plot_paths["Categorical duration bins"])

    settings_df.to_csv(output_dir / "final_clustering_setting_for_report.csv", index=False)
    profiles.to_csv(output_dir / PROFILE_CSV.name, index=False)
    write_final_cluster_profile_report(
        settings_df=settings_df,
        profiles=profiles,
        cat_candidates=cat_candidates,
        dog_candidates=dog_candidates,
        plot_paths=plot_paths,
        report_path=output_dir / PROFILE_MD.name,
    )
    return settings_df, profiles, clustered
