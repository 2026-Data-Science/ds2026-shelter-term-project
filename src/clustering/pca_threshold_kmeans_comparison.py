from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.clustering.pca_kmeans_comparison import (
    DEFAULT_K_RANGE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    OUTPUT_DIR,
    _as_dense,
    _select_n_components_for_variance,
)
from src.clustering.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    build_long_stay_preprocessor,
)

THRESHOLD_CSV = OUTPUT_DIR / "pca_threshold_kmeans_comparison.csv"
THRESHOLD_SUMMARY_MD = OUTPUT_DIR / "pca_threshold_kmeans_summary.md"

PCA_THRESHOLDS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
CLUSTER_IMBALANCE_RATIO_THRESHOLD = 0.05


def _silhouette_sample_size(n_samples: int) -> int | None:
    if n_samples > DEFAULT_SILHOUETTE_SAMPLE_SIZE:
        return DEFAULT_SILHOUETTE_SAMPLE_SIZE
    return None


def _top_value(series: pd.Series) -> str:
    cleaned = series.fillna("Unknown").astype(str)
    if cleaned.empty:
        return "Unknown (0.0%)"
    counts = cleaned.value_counts(dropna=False)
    value = counts.index[0]
    ratio = counts.iloc[0] / len(cleaned) * 100.0
    return f"{value} ({ratio:.1f}%)"


def _species_subset(clustering_frame: pd.DataFrame, species: str) -> pd.DataFrame:
    return clustering_frame[clustering_frame["AnimalType"].astype(str) == species].copy()


def _fit_encoded(species_frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    preprocessor = build_long_stay_preprocessor()
    encoded = _as_dense(
        preprocessor.fit_transform(species_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    )
    return encoded, encoded.shape[1]


def _prepare_pca_reduced(
    encoded: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int, float]:
    pca_full = PCA(n_components=encoded.shape[1], svd_solver="full", random_state=DEFAULT_RANDOM_STATE)
    full_reduced = pca_full.fit_transform(encoded)
    explained = pca_full.explained_variance_ratio_
    n_components = _select_n_components_for_variance(explained, threshold)
    reduced = full_reduced[:, :n_components]
    evr_sum = float(explained[:n_components].sum())
    return reduced, n_components, evr_sum


def _kmeans_fit(reduced: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    model = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE, n_init=20)
    labels = model.fit_predict(reduced)
    return labels, float(model.inertia_)


def _dual_silhouette(
    reduced: np.ndarray,
    encoded: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    sample_size = _silhouette_sample_size(len(reduced))
    sil_pca = float(
        silhouette_score(
            reduced,
            labels,
            sample_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE,
        )
    )
    sil_full = float(
        silhouette_score(
            encoded,
            labels,
            sample_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE,
        )
    )
    return sil_pca, sil_full


def _cluster_size_metrics(labels: np.ndarray) -> dict[str, object]:
    _, counts = np.unique(labels, return_counts=True)
    ordered = sorted(int(value) for value in counts)
    total = int(labels.shape[0])
    ratios = [value / total for value in ordered]
    summary = "|".join(str(value) for value in ordered)
    return {
        "cluster_size_summary": summary,
        "min_cluster_size": min(ordered),
        "max_cluster_size": max(ordered),
        "min_cluster_ratio": round(min(ratios), 6),
        "max_cluster_ratio": round(max(ratios), 6),
    }


def _duration_profile_table(
    species_frame: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    frame = species_frame.reset_index(drop=True).copy()
    frame["cluster"] = labels
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(frame["cluster"].unique()):
        subset = frame[frame["cluster"] == cluster_id]
        duration = subset["length_of_stay_days"]
        rows.append(
            {
                "cluster": int(cluster_id),
                "count": int(len(subset)),
                "duration_mean_days": round(float(duration.mean()), 2),
                "duration_median_days": round(float(duration.median()), 2),
                "long_stay_30_ratio_pct": round(float(subset["long_stay_30"].mean()) * 100.0, 2),
                "long_stay_60_ratio_pct": round(float(subset["long_stay_60"].mean()) * 100.0, 2),
                "top_intake_type": _top_value(subset["intake_type"]),
                "top_neuter_status": _top_value(subset["neuter_status"]),
                "top_intake_age_group": _top_value(subset["intake_age_group"]),
                "top_primary_breed": _top_value(subset["primary_breed"]),
            }
        )
    return pd.DataFrame(rows)


def _build_comparison_rows(
    clustering_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for species in ["Cat", "Dog"]:
        species_frame = _species_subset(clustering_frame, species)
        encoded, encoded_dimensions = _fit_encoded(species_frame)

        configs: list[tuple[str, str | float, np.ndarray, int, float]] = [
            ("baseline", "none", encoded, encoded_dimensions, 1.0),
        ]
        for threshold in PCA_THRESHOLDS:
            reduced, n_components, evr_sum = _prepare_pca_reduced(encoded, threshold)
            configs.append(
                ("pca_variance", threshold, reduced, n_components, evr_sum),
            )

        for method, pca_threshold, reduced, n_components, evr_sum in configs:
            for k in DEFAULT_K_RANGE:
                labels, inertia = _kmeans_fit(reduced, k)
                sil_pca, sil_full = _dual_silhouette(reduced, encoded, labels)
                size_metrics = _cluster_size_metrics(labels)
                rows.append(
                    {
                        "species": species,
                        "method": method,
                        "pca_threshold": (
                            "none" if method == "baseline" else f"{float(pca_threshold):.2f}"
                        ),
                        "n_components_selected": int(n_components),
                        "explained_variance_ratio_sum": round(float(evr_sum), 6),
                        "encoded_dimensions": int(encoded_dimensions),
                        "k": int(k),
                        "inertia": inertia,
                        "silhouette_pca_space": round(sil_pca, 6),
                        "silhouette_full_space": round(sil_full, 6),
                        "silhouette_gap_pca_minus_full": round(sil_pca - sil_full, 6),
                        **size_metrics,
                    }
                )

    return pd.DataFrame(rows)


def _threshold_summary_rows(
    comparison: pd.DataFrame,
    clustering_frame: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []

    for species in sorted(comparison["species"].unique()):
        species_frame = _species_subset(clustering_frame, species)
        encoded, _ = _fit_encoded(species_frame)

        baseline = comparison[
            (comparison["species"] == species) & (comparison["method"] == "baseline")
        ]
        baseline_best = baseline.sort_values(
            by=["silhouette_full_space", "k"],
            ascending=[False, True],
        ).iloc[0]

        for threshold in PCA_THRESHOLDS:
            subset = comparison[
                (comparison["species"] == species)
                & (comparison["method"] == "pca_variance")
                & (comparison["pca_threshold"] == f"{threshold:.2f}")
            ]
            if subset.empty:
                continue

            n_components = int(subset["n_components_selected"].iloc[0])
            evr_sum = float(subset["explained_variance_ratio_sum"].iloc[0])

            best_pca_row = subset.sort_values(
                by=["silhouette_pca_space", "k"],
                ascending=[False, True],
            ).iloc[0]
            best_full_row = subset.sort_values(
                by=["silhouette_full_space", "k"],
                ascending=[False, True],
            ).iloc[0]

            stable_full = subset[
                (subset["min_cluster_size"] / subset["max_cluster_size"])
                >= CLUSTER_IMBALANCE_RATIO_THRESHOLD
            ]
            if stable_full.empty:
                stable_full = subset
            balanced_full_best = stable_full.sort_values(
                by=["silhouette_full_space", "k"],
                ascending=[False, True],
            ).iloc[0]

            reduced, _, _ = _prepare_pca_reduced(encoded, threshold)
            labels, _ = _kmeans_fit(reduced, int(balanced_full_best["k"]))
            profile = _duration_profile_table(species_frame, labels)
            duration_spread = float(profile["duration_mean_days"].max() - profile["duration_mean_days"].min())

            full_improves = float(balanced_full_best["silhouette_full_space"]) > float(
                baseline_best["silhouette_full_space"]
            )
            gap = float(best_pca_row["silhouette_gap_pca_minus_full"])
            dominant_axis_risk = gap >= 0.15 and float(best_pca_row["silhouette_full_space"]) < float(
                baseline_best["silhouette_full_space"]
            )

            if dominant_axis_risk:
                recommendation = (
                    f"Not recommended: large PCA/full gap ({gap:.3f}) and full-space silhouette "
                    f"below baseline. Prefer baseline k={int(baseline_best['k'])} for reporting."
                )
            elif full_improves:
                recommendation = (
                    f"Candidate for discussion: threshold={threshold:.2f}, k={int(balanced_full_best['k'])}, "
                    f"full-space silhouette={float(balanced_full_best['silhouette_full_space']):.4f} "
                    f"(baseline={float(baseline_best['silhouette_full_space']):.4f})."
                )
            else:
                recommendation = (
                    f"Keep baseline for reporting: threshold={threshold:.2f} does not beat baseline "
                    f"full-space silhouette ({float(baseline_best['silhouette_full_space']):.4f})."
                )

            summary_rows.append(
                {
                    "species": species,
                    "pca_threshold": f"{threshold:.2f}",
                    "n_components_selected": n_components,
                    "explained_variance_ratio_sum": round(evr_sum, 6),
                    "best_k_by_pca_silhouette": int(best_pca_row["k"]),
                    "best_pca_silhouette": float(best_pca_row["silhouette_pca_space"]),
                    "corresponding_full_space_silhouette": float(
                        best_pca_row["silhouette_full_space"]
                    ),
                    "best_k_by_full_space_silhouette": int(best_full_row["k"]),
                    "best_full_space_silhouette": float(best_full_row["silhouette_full_space"]),
                    "balanced_best_k_full_space": int(balanced_full_best["k"]),
                    "balanced_full_space_silhouette": float(
                        balanced_full_best["silhouette_full_space"]
                    ),
                    "duration_mean_spread_days": duration_spread,
                    "baseline_best_k": int(baseline_best["k"]),
                    "baseline_best_full_space_silhouette": float(
                        baseline_best["silhouette_full_space"]
                    ),
                    "recommendation": recommendation,
                }
            )

    return pd.DataFrame(summary_rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_No data._"
    display = frame[columns] if columns else frame
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def write_pca_threshold_kmeans_summary(
    comparison: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    report_path: Path = THRESHOLD_SUMMARY_MD,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# PCA Threshold K-Means Comparison")
    lines.append("")
    lines.append(
        "Variance-threshold PCA (0.70–0.95) vs baseline K-Means on the full encoded feature matrix. "
        "Production clustering outputs are unchanged."
    )
    lines.append("")
    lines.append(f"- Detailed CSV: `{THRESHOLD_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    lines.append("## Threshold vs Selected Components")
    lines.append("")
    for species in ["Cat", "Dog"]:
        subset = threshold_summary[threshold_summary["species"] == species]
        lines.append(f"### {species}")
        lines.append("")
        lines.append(
            _markdown_table(
                subset[
                    [
                        "pca_threshold",
                        "n_components_selected",
                        "explained_variance_ratio_sum",
                        "balanced_best_k_full_space",
                        "balanced_full_space_silhouette",
                        "baseline_best_full_space_silhouette",
                    ]
                ]
            )
        )
        lines.append("")

    lines.append("## PCA-Space vs Full-Space Silhouette Gap")
    lines.append("")
    pca_rows = comparison[comparison["method"] == "pca_variance"].copy()
    gap_summary = (
        pca_rows.groupby(["species", "pca_threshold"], as_index=False)
        .agg(
            mean_gap=("silhouette_gap_pca_minus_full", "mean"),
            max_gap=("silhouette_gap_pca_minus_full", "max"),
            mean_full=("silhouette_full_space", "mean"),
            mean_pca=("silhouette_pca_space", "mean"),
        )
        .sort_values(["species", "pca_threshold"], ascending=[True, False])
    )
    lines.append(_markdown_table(gap_summary))
    lines.append("")
    lines.append(
        "Lower thresholds reduce `n_components` and often widen the PCA/full silhouette gap."
    )
    lines.append("")

    lines.append("## Threshold Best-Result Summary")
    lines.append("")
    lines.append(_markdown_table(threshold_summary))
    lines.append("")

    lines.append("## Baseline vs Threshold Improvement")
    lines.append("")
    for species in ["Cat", "Dog"]:
        subset = threshold_summary[threshold_summary["species"] == species]
        baseline_score = float(subset["baseline_best_full_space_silhouette"].iloc[0])
        improves = subset[
            subset["balanced_full_space_silhouette"] > baseline_score
        ]
        lines.append(f"### {species}")
        lines.append("")
        lines.append(f"- Baseline best full-space silhouette: **{baseline_score:.4f}**")
        if improves.empty:
            lines.append("- No threshold beats baseline on balanced full-space silhouette.")
        else:
            lines.append("- Thresholds beating baseline (balanced full-space):")
            for _, row in improves.iterrows():
                lines.append(
                    f"  - threshold={row['pca_threshold']}, k={int(row['balanced_best_k_full_space'])}, "
                    f"silhouette={float(row['balanced_full_space_silhouette']):.4f}, "
                    f"n_components={int(row['n_components_selected'])}"
                )
        lines.append("")

    lines.append("## Report Recommendations (not auto-applied)")
    lines.append("")
    for species in ["Cat", "Dog"]:
        subset = threshold_summary[threshold_summary["species"] == species]
        lines.append(f"### {species}")
        lines.append("")
        for _, row in subset.iterrows():
            lines.append(f"- **threshold {row['pca_threshold']}**: {row['recommendation']}")
        lines.append("")

    production_threshold = "0.95"
    lines.append("### Production note")
    lines.append("")
    lines.append(
        f"- Current production workflow uses PCA variance threshold **{production_threshold}** "
        "with k selected from silhouette in PCA space."
    )
    lines.append(
        "- This experiment suggests evaluating thresholds using **full-space silhouette**, "
        "**cluster balance**, and **duration/long-stay profile spread**, not PCA-space silhouette alone."
    )
    lines.append(
        "- **Do not replace production clustering automatically** from this file; use it for report discussion only."
    )
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("- **최종 production clustering은 변경하지 않았습니다.**")
    lines.append("- **classification 파이프라인은 변경하지 않았습니다.**")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_pca_threshold_kmeans_comparison(
    clustering_frame: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare baseline vs PCA variance-threshold K-Means without changing production outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = _build_comparison_rows(clustering_frame)
    threshold_summary = _threshold_summary_rows(comparison, clustering_frame)

    comparison.to_csv(output_dir / THRESHOLD_CSV.name, index=False)
    write_pca_threshold_kmeans_summary(
        comparison=comparison,
        threshold_summary=threshold_summary,
        report_path=output_dir / THRESHOLD_SUMMARY_MD.name,
    )
    return comparison, threshold_summary
