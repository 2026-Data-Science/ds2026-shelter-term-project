from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.clustering.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    build_long_stay_preprocessor,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "long_stay_risk_clustering"
COMPARISON_CSV = OUTPUT_DIR / "pca_kmeans_comparison.csv"
COMPARISON_MD = OUTPUT_DIR / "pca_kmeans_comparison.md"

DEFAULT_K_RANGE = (2, 3, 4, 5, 6)
DEFAULT_RANDOM_STATE = 42
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 3000

VARIANCE_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
FIXED_N_COMPONENTS = (2, 3, 5, 8, 10, 15, 20)
CLUSTER_IMBALANCE_RATIO_THRESHOLD = 0.05


def _as_dense(matrix: object) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _silhouette_sample_size(n_samples: int) -> int | None:
    if n_samples > DEFAULT_SILHOUETTE_SAMPLE_SIZE:
        return DEFAULT_SILHOUETTE_SAMPLE_SIZE
    return None


def _cluster_size_fields(labels: np.ndarray) -> tuple[str, int, int]:
    _, counts = np.unique(labels, return_counts=True)
    ordered = sorted(int(value) for value in counts)
    summary = "|".join(str(value) for value in ordered)
    return summary, min(ordered), max(ordered)


def _select_n_components_for_variance(
    explained: np.ndarray,
    threshold: float,
) -> int:
    cumulative = np.cumsum(explained)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def _score_kmeans_grid(
    reduced: np.ndarray,
    species: str,
    method: str,
    pca_setting: str,
    n_components_selected: int,
    explained_variance_ratio_sum: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample_size = _silhouette_sample_size(len(reduced))
    for k in DEFAULT_K_RANGE:
        model = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE, n_init=20)
        labels = model.fit_predict(reduced)
        score = float(
            silhouette_score(
                reduced,
                labels,
                sample_size=sample_size,
                random_state=DEFAULT_RANDOM_STATE,
            )
        )
        size_summary, min_size, max_size = _cluster_size_fields(labels)
        rows.append(
            {
                "species": species,
                "method": method,
                "pca_setting": pca_setting,
                "n_components_selected": int(n_components_selected),
                "explained_variance_ratio_sum": round(float(explained_variance_ratio_sum), 6),
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette_score": score,
                "cluster_size_summary": size_summary,
                "min_cluster_size": int(min_size),
                "max_cluster_size": int(max_size),
            }
        )
    return rows


def _iter_embedding_configs(encoded: np.ndarray) -> list[tuple[str, str, np.ndarray, int, float]]:
    """Yield (method, pca_setting, reduced_matrix, n_components, evr_sum)."""
    max_dim = encoded.shape[1]
    configs: list[tuple[str, str, np.ndarray, int, float]] = []

    configs.append(
        (
            "baseline",
            "none",
            encoded,
            max_dim,
            1.0,
        )
    )

    pca_full = PCA(n_components=max_dim, svd_solver="full", random_state=DEFAULT_RANDOM_STATE)
    full_reduced = pca_full.fit_transform(encoded)
    explained = pca_full.explained_variance_ratio_

    for threshold in VARIANCE_THRESHOLDS:
        n_components = _select_n_components_for_variance(explained, threshold)
        reduced = full_reduced[:, :n_components]
        configs.append(
            (
                "pca_variance",
                f"threshold={threshold:.2f}",
                reduced,
                n_components,
                float(explained[:n_components].sum()),
            )
        )

    for n_components in FIXED_N_COMPONENTS:
        if n_components > max_dim:
            continue
        pca = PCA(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
        reduced = pca.fit_transform(encoded)
        configs.append(
            (
                "pca_fixed",
                f"n_components={n_components}",
                reduced,
                n_components,
                float(pca.explained_variance_ratio_.sum()),
            )
        )

    return configs


def run_pca_kmeans_comparison(
    clustering_frame: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Compare baseline vs PCA-reduced K-Means without changing the production workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for species in ["Cat", "Dog"]:
        subset = clustering_frame[clustering_frame["AnimalType"].astype(str) == species].copy()
        if subset.empty:
            continue

        preprocessor = build_long_stay_preprocessor()
        encoded = _as_dense(preprocessor.fit_transform(subset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]))

        for method, pca_setting, reduced, n_components, evr_sum in _iter_embedding_configs(encoded):
            all_rows.extend(
                _score_kmeans_grid(
                    reduced=reduced,
                    species=species,
                    method=method,
                    pca_setting=pca_setting,
                    n_components_selected=n_components,
                    explained_variance_ratio_sum=evr_sum,
                )
            )

    comparison = pd.DataFrame(all_rows)
    comparison.to_csv(output_dir / COMPARISON_CSV.name, index=False)
    write_pca_kmeans_comparison_report(comparison, output_dir / COMPARISON_MD.name)
    return comparison


def _best_row_per_species(frame: pd.DataFrame, species: str) -> pd.Series:
    subset = frame[frame["species"] == species]
    return subset.sort_values(
        by=["silhouette_score", "k"],
        ascending=[False, True],
    ).iloc[0]


def _baseline_best(frame: pd.DataFrame, species: str) -> pd.Series:
    subset = frame[(frame["species"] == species) & (frame["method"] == "baseline")]
    return subset.sort_values(
        by=["silhouette_score", "k"],
        ascending=[False, True],
    ).iloc[0]


def _imbalanced_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ratio = frame["min_cluster_size"] / frame["max_cluster_size"].replace(0, np.nan)
    return frame[ratio < CLUSTER_IMBALANCE_RATIO_THRESHOLD].copy()


def write_pca_kmeans_comparison_report(
    comparison: pd.DataFrame,
    report_path: Path = COMPARISON_MD,
) -> Path:
    """Write markdown summary for PCA vs baseline K-Means comparison."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# PCA vs Baseline K-Means Comparison")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This experiment compares clustering quality when K-Means is applied to the "
        "full preprocessed feature matrix (baseline) versus PCA-reduced embeddings. "
        "The production long-stay clustering workflow is unchanged."
    )
    lines.append("")
    lines.append("## Experiment Design")
    lines.append("")
    lines.append("- Species: Cat and Dog (evaluated separately)")
    lines.append("- Baseline: K-Means on full encoded features (no PCA before clustering)")
    lines.append("- PCA variance thresholds: 0.80, 0.85, 0.90, 0.95")
    lines.append("- PCA fixed `n_components`: 2, 3, 5, 8, 10, 15, 20 (skipped when > feature dimension)")
    lines.append("- K range: 2 through 6 for every configuration")
    lines.append(
        f"- Silhouette: sample size capped at {DEFAULT_SILHOUETTE_SAMPLE_SIZE} when needed"
    )
    lines.append("")
    lines.append(f"- Full results CSV: `{COMPARISON_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append("")

    recommendations: list[str] = []

    for species in sorted(comparison["species"].unique()):
        species_frame = comparison[comparison["species"] == species]
        best_overall = _best_row_per_species(species_frame, species)
        best_baseline = _baseline_best(species_frame, species)
        best_pca = species_frame[species_frame["method"] != "baseline"].sort_values(
            by=["silhouette_score", "k"],
            ascending=[False, True],
        ).iloc[0]

        improved = float(best_pca["silhouette_score"]) > float(best_baseline["silhouette_score"])
        imbalanced = _imbalanced_rows(species_frame)

        lines.append(f"## {species}")
        lines.append("")
        lines.append("### Best silhouette overall")
        lines.append("")
        lines.append(
            f"- method=`{best_overall['method']}`, pca_setting=`{best_overall['pca_setting']}`, "
            f"k={int(best_overall['k'])}, silhouette={float(best_overall['silhouette_score']):.4f}, "
            f"n_components={int(best_overall['n_components_selected'])}, "
            f"cluster sizes=`{best_overall['cluster_size_summary']}`"
        )
        lines.append("")
        lines.append("### Best baseline (no PCA before K-Means)")
        lines.append("")
        lines.append(
            f"- k={int(best_baseline['k'])}, silhouette={float(best_baseline['silhouette_score']):.4f}, "
            f"n_components={int(best_baseline['n_components_selected'])}, "
            f"cluster sizes=`{best_baseline['cluster_size_summary']}`"
        )
        lines.append("")
        lines.append("### Best PCA-reduced K-Means")
        lines.append("")
        lines.append(
            f"- method=`{best_pca['method']}`, pca_setting=`{best_pca['pca_setting']}`, "
            f"k={int(best_pca['k'])}, silhouette={float(best_pca['silhouette_score']):.4f}, "
            f"n_components={int(best_pca['n_components_selected'])}, "
            f"cluster sizes=`{best_pca['cluster_size_summary']}`"
        )
        lines.append("")
        lines.append("### Baseline vs PCA-reduced")
        lines.append("")
        if improved:
            delta = float(best_pca["silhouette_score"]) - float(best_baseline["silhouette_score"])
            lines.append(
                f"- PCA-reduced K-Means improves over baseline by **{delta:.4f}** silhouette (best-of-grid)."
            )
        else:
            delta = float(best_baseline["silhouette_score"]) - float(best_pca["silhouette_score"])
            lines.append(
                f"- Baseline is better than the best PCA-reduced setting by **{delta:.4f}** silhouette."
            )
        lines.append("")
        lines.append("### Severely imbalanced cluster sizes")
        lines.append("")
        lines.append(
            f"- Flagged when `min_cluster_size / max_cluster_size < {CLUSTER_IMBALANCE_RATIO_THRESHOLD}`."
        )
        if imbalanced.empty:
            lines.append("- No combinations flagged under this rule.")
        else:
            lines.append(f"- {len(imbalanced)} combinations flagged (see CSV for details).")
            preview = imbalanced.sort_values("min_cluster_size").head(5)
            for _, row in preview.iterrows():
                lines.append(
                    f"  - `{row['method']}` / `{row['pca_setting']}` / k={int(row['k'])}: "
                    f"sizes=`{row['cluster_size_summary']}`"
                )
        lines.append("")

        rec_method = str(best_overall["method"])
        rec_setting = str(best_overall["pca_setting"])
        rec_k = int(best_overall["k"])
        rec_sil = float(best_overall["silhouette_score"])
        min_max_ratio = float(best_overall["min_cluster_size"]) / float(best_overall["max_cluster_size"])
        if min_max_ratio < CLUSTER_IMBALANCE_RATIO_THRESHOLD:
            stable = species_frame[
                (species_frame["min_cluster_size"] / species_frame["max_cluster_size"])
                >= CLUSTER_IMBALANCE_RATIO_THRESHOLD
            ]
            if not stable.empty:
                stable_best = stable.sort_values(
                    by=["silhouette_score", "k"],
                    ascending=[False, True],
                ).iloc[0]
                rec_method = str(stable_best["method"])
                rec_setting = str(stable_best["pca_setting"])
                rec_k = int(stable_best["k"])
                rec_sil = float(stable_best["silhouette_score"])
                note = (
                    f"best silhouette combo was imbalanced; recommend `{rec_method}` / "
                    f"`{rec_setting}` / k={rec_k} (silhouette={rec_sil:.4f}) as a more balanced alternative"
                )
            else:
                note = (
                    f"recommend reviewing `{rec_method}` / `{rec_setting}` / k={rec_k} "
                    f"(silhouette={rec_sil:.4f}) but cluster balance is weak across the grid"
                )
        else:
            note = (
                f"recommend `{rec_method}` / `{rec_setting}` / k={rec_k} "
                f"(silhouette={rec_sil:.4f}) for report discussion (not applied to production)"
            )
        recommendations.append(f"- **{species}**: {note}")

    lines.append("## Recommendations For Final Report (not auto-applied)")
    lines.append("")
    lines.extend(recommendations)
    lines.append("")
    lines.append("## Top 10 Combinations Per Species")
    lines.append("")
    for species in sorted(comparison["species"].unique()):
        lines.append(f"### {species}")
        lines.append("")
        top = (
            comparison[comparison["species"] == species]
            .sort_values(by=["silhouette_score", "k"], ascending=[False, True])
            .head(10)
        )
        display = top[
            [
                "method",
                "pca_setting",
                "n_components_selected",
                "explained_variance_ratio_sum",
                "k",
                "silhouette_score",
                "cluster_size_summary",
            ]
        ]
        lines.append("| method | pca_setting | n_components | evr_sum | k | silhouette | cluster_sizes |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in display.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method"]),
                        str(row["pca_setting"]),
                        str(int(row["n_components_selected"])),
                        f"{float(row['explained_variance_ratio_sum']):.4f}",
                        str(int(row["k"])),
                        f"{float(row['silhouette_score']):.4f}",
                        str(row["cluster_size_summary"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
