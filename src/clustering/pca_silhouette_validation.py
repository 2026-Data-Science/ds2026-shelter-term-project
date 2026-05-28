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
    VARIANCE_THRESHOLDS,
    _as_dense,
    _select_n_components_for_variance,
)
from src.clustering.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    build_long_stay_preprocessor,
)

VALIDATION_CSV = OUTPUT_DIR / "pca_silhouette_validation.csv"
VALIDATION_MD = OUTPUT_DIR / "pca_silhouette_validation.md"

FOCUS_PCA_SETTING = "n_components=2"
FOCUS_DOG_K = 4
FOCUS_BASELINE_DOG_K = 6


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


def _fit_encoded(species_frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = build_long_stay_preprocessor()
    feature_frame = species_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    encoded = _as_dense(preprocessor.fit_transform(feature_frame))
    feature_names = list(preprocessor.get_feature_names_out())
    return encoded, feature_names


def _fit_pca_fixed(encoded: np.ndarray, n_components: int) -> tuple[np.ndarray, float, PCA]:
    pca = PCA(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
    reduced = pca.fit_transform(encoded)
    return reduced, float(pca.explained_variance_ratio_.sum()), pca


def _fit_labels(
    encoded: np.ndarray,
    method: str,
    pca_setting: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (reduced_for_clustering, labels, explained_variance_ratio_sum)."""
    if method == "baseline":
        reduced = encoded
        return reduced, _kmeans_labels(reduced, k), 1.0

    if method == "pca_fixed":
        n_components = int(pca_setting.split("=")[1])
        reduced, evr_sum, _ = _fit_pca_fixed(encoded, n_components)
        return reduced, _kmeans_labels(reduced, k), evr_sum

    if method == "pca_variance":
        threshold = float(pca_setting.split("=")[1])
        pca_full = PCA(n_components=encoded.shape[1], svd_solver="full", random_state=DEFAULT_RANDOM_STATE)
        full_reduced = pca_full.fit_transform(encoded)
        explained = pca_full.explained_variance_ratio_
        n_components = _select_n_components_for_variance(explained, threshold)
        reduced = full_reduced[:, :n_components]
        evr_sum = float(explained[:n_components].sum())
        return reduced, _kmeans_labels(reduced, k), evr_sum

    raise ValueError(f"Unsupported method: {method}")


def _kmeans_labels(reduced: np.ndarray, k: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE, n_init=20)
    return model.fit_predict(reduced)


def _dual_silhouette(
    reduced: np.ndarray,
    encoded: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    sample_size = _silhouette_sample_size(len(reduced))
    silhouette_pca = float(
        silhouette_score(
            reduced,
            labels,
            sample_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE,
        )
    )
    silhouette_full = float(
        silhouette_score(
            encoded,
            labels,
            sample_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE,
        )
    )
    return silhouette_pca, silhouette_full


def _cluster_size_table(frame: pd.DataFrame) -> pd.DataFrame:
    total = len(frame)
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(frame["cluster"].unique()):
        count = int((frame["cluster"] == cluster_id).sum())
        rows.append(
            {
                "record_type": "cluster_size",
                "cluster": int(cluster_id),
                "count": count,
                "ratio_pct": round(count / total * 100.0, 2),
            }
        )
    return pd.DataFrame(rows)


def _cluster_profile_table(frame: pd.DataFrame, config_label: str) -> pd.DataFrame:
    total = len(frame)
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(frame["cluster"].unique()):
        subset = frame[frame["cluster"] == cluster_id]
        duration = subset["length_of_stay_days"]
        rows.append(
            {
                "record_type": "cluster_profile",
                "config_label": config_label,
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
                "long_stay_30_ratio_pct": round(float(subset["long_stay_30"].mean()) * 100.0, 2),
                "long_stay_60_ratio_pct": round(float(subset["long_stay_60"].mean()) * 100.0, 2),
            }
        )
    return pd.DataFrame(rows)


def _pc_loadings_table(
    pca: PCA,
    feature_names: list[str],
    species: str,
    top_n: int = 10,
) -> pd.DataFrame:
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=feature_names,
    )
    rows: list[dict[str, object]] = []
    for component in ["PC1", "PC2"]:
        ranked = loadings[component].abs().sort_values(ascending=False).head(top_n)
        for feature, loading in ranked.items():
            signed = float(loadings.loc[feature, component])
            rows.append(
                {
                    "record_type": "pc_loading",
                    "species": species,
                    "component": component,
                    "feature": feature,
                    "loading": round(signed, 4),
                    "abs_loading": round(abs(signed), 4),
                }
            )
    return pd.DataFrame(rows)


def _build_silhouette_dual_rows(clustering_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for species in ["Cat", "Dog"]:
        species_frame = _species_subset(clustering_frame, species)
        encoded, _ = _fit_encoded(species_frame)

        configs: list[tuple[str, str]] = [("baseline", "none")]
        configs += [("pca_variance", setting) for setting in [f"threshold={v:.2f}" for v in VARIANCE_THRESHOLDS]]
        configs += [("pca_fixed", f"n_components={n}") for n in (2, 3, 5, 8, 10, 15, 20) if n <= encoded.shape[1]]

        for method, pca_setting in configs:
            for k in DEFAULT_K_RANGE:
                reduced, labels, evr_sum = _fit_labels(encoded, method, pca_setting, k)
                sil_pca, sil_full = _dual_silhouette(reduced, encoded, labels)
                rows.append(
                    {
                        "record_type": "silhouette_dual",
                        "species": species,
                        "method": method,
                        "pca_setting": pca_setting,
                        "k": int(k),
                        "silhouette_pca_space": round(sil_pca, 6),
                        "silhouette_full_space": round(sil_full, 6),
                        "silhouette_gap_pca_minus_full": round(sil_pca - sil_full, 6),
                        "explained_variance_ratio_sum": round(evr_sum, 6),
                    }
                )
    return pd.DataFrame(rows)


def _labeled_frame(
    species_frame: pd.DataFrame,
    method: str,
    pca_setting: str,
    k: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, PCA | None, list[str]]:
    encoded, feature_names = _fit_encoded(species_frame)
    reduced, labels, evr_sum = _fit_labels(encoded, method, pca_setting, k)
    frame = species_frame.reset_index(drop=True).copy()
    frame["cluster"] = labels

    pca_model: PCA | None = None
    if method == "pca_fixed":
        _, _, pca_model = _fit_pca_fixed(encoded, int(pca_setting.split("=")[1]))

    return frame, encoded, reduced, evr_sum, pca_model, feature_names


def run_pca_silhouette_validation(
    clustering_frame: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Validate suspicious PCA-space silhouette scores without changing production clustering."""
    output_dir.mkdir(parents=True, exist_ok=True)

    evr_rows: list[dict[str, object]] = []
    for species in ["Cat", "Dog"]:
        encoded, _ = _fit_encoded(_species_subset(clustering_frame, species))
        _, evr_sum, _ = _fit_pca_fixed(encoded, 2)
        evr_rows.append(
            {
                "record_type": "evr_n2",
                "species": species,
                "pca_setting": FOCUS_PCA_SETTING,
                "n_components_selected": 2,
                "explained_variance_ratio_sum": round(evr_sum, 6),
                "explained_variance_pct": round(evr_sum * 100.0, 2),
                "encoded_dimensions": int(encoded.shape[1]),
            }
        )
    evr_df = pd.DataFrame(evr_rows)

    silhouette_df = _build_silhouette_dual_rows(clustering_frame)

    dog_frame, _, _, dog_evr, dog_pca, dog_features = _labeled_frame(
        _species_subset(clustering_frame, "Dog"),
        "pca_fixed",
        FOCUS_PCA_SETTING,
        FOCUS_DOG_K,
    )
    dog_sizes = _cluster_size_table(dog_frame)
    dog_sizes["species"] = "Dog"
    dog_sizes["method"] = "pca_fixed"
    dog_sizes["pca_setting"] = FOCUS_PCA_SETTING
    dog_sizes["k"] = FOCUS_DOG_K

    dog_profiles = _cluster_profile_table(
        dog_frame,
        config_label="Dog pca_fixed n_components=2 k=4",
    )

    baseline_frame, _, _, _, _, _ = _labeled_frame(
        _species_subset(clustering_frame, "Dog"),
        "baseline",
        "none",
        FOCUS_BASELINE_DOG_K,
    )
    baseline_profiles = _cluster_profile_table(
        baseline_frame,
        config_label="Dog baseline k=6",
    )

    loadings_df = pd.DataFrame()
    if dog_pca is not None:
        loadings_df = _pc_loadings_table(dog_pca, dog_features, "Dog")

    combined = pd.concat(
        [
            evr_df,
            silhouette_df,
            dog_sizes,
            dog_profiles,
            baseline_profiles,
            loadings_df,
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(output_dir / VALIDATION_CSV.name, index=False)
    write_pca_silhouette_validation_report(
        evr_df=evr_df,
        silhouette_df=silhouette_df,
        dog_sizes=dog_sizes,
        dog_profiles=dog_profiles,
        baseline_profiles=baseline_profiles,
        loadings_df=loadings_df,
        dog_evr=dog_evr,
        report_path=output_dir / VALIDATION_MD.name,
    )
    return combined


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


def write_pca_silhouette_validation_report(
    evr_df: pd.DataFrame,
    silhouette_df: pd.DataFrame,
    dog_sizes: pd.DataFrame,
    dog_profiles: pd.DataFrame,
    baseline_profiles: pd.DataFrame,
    loadings_df: pd.DataFrame,
    dog_evr: float,
    report_path: Path = VALIDATION_MD,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# PCA Silhouette Validation")
    lines.append("")
    lines.append(
        "This report checks whether high silhouette scores in 2D PCA space are "
        "trustworthy when evaluated in the original encoded feature space. "
        "Production clustering settings are not changed."
    )
    lines.append("")
    lines.append(f"- CSV: `{VALIDATION_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append("")

    lines.append("## 1. Explained Variance For `pca_fixed n_components=2`")
    lines.append("")
    lines.append(_markdown_table(evr_df))
    lines.append("")
    lines.append(
        f"- Dog 2D PCA explains **{dog_evr * 100:.2f}%** of encoded feature variance "
        f"({dog_evr:.4f} cumulative explained variance ratio)."
    )
    lines.append(
        "- A high silhouette in PCA space can be inflated when clustering happens in a "
        "low-dimensional projection that discards most feature variation."
    )
    lines.append("")

    lines.append("## 2. Silhouette In PCA Space vs Full Encoded Space")
    lines.append("")
    lines.append(
        "Same K-Means labels are used; only the distance space changes for silhouette calculation."
    )
    lines.append("")

    focus = silhouette_df[
        (silhouette_df["species"] == "Dog")
        & (silhouette_df["method"] == "pca_fixed")
        & (silhouette_df["pca_setting"] == FOCUS_PCA_SETTING)
        & (silhouette_df["k"] == FOCUS_DOG_K)
    ]
    if not focus.empty:
        row = focus.iloc[0]
        lines.append("### Focus: Dog `pca_fixed n_components=2`, k=4")
        lines.append("")
        lines.append(
            f"- silhouette_pca_space = **{row['silhouette_pca_space']:.4f}**"
        )
        lines.append(
            f"- silhouette_full_space = **{row['silhouette_full_space']:.4f}**"
        )
        lines.append(
            f"- gap (PCA - full) = **{row['silhouette_gap_pca_minus_full']:.4f}**"
        )
        lines.append("")

    dog_n2 = silhouette_df[
        (silhouette_df["species"] == "Dog")
        & (silhouette_df["method"] == "pca_fixed")
        & (silhouette_df["pca_setting"] == FOCUS_PCA_SETTING)
    ].sort_values("k")
    lines.append("### Dog `pca_fixed n_components=2` Across k")
    lines.append("")
    lines.append(
        _markdown_table(
            dog_n2[
                [
                    "k",
                    "silhouette_pca_space",
                    "silhouette_full_space",
                    "silhouette_gap_pca_minus_full",
                    "explained_variance_ratio_sum",
                ]
            ]
        )
    )
    lines.append("")

    cat_n2 = silhouette_df[
        (silhouette_df["species"] == "Cat")
        & (silhouette_df["method"] == "pca_fixed")
        & (silhouette_df["pca_setting"] == FOCUS_PCA_SETTING)
    ].sort_values("k")
    lines.append("### Cat `pca_fixed n_components=2` Across k")
    lines.append("")
    lines.append(
        _markdown_table(
            cat_n2[
                [
                    "k",
                    "silhouette_pca_space",
                    "silhouette_full_space",
                    "silhouette_gap_pca_minus_full",
                    "explained_variance_ratio_sum",
                ]
            ]
        )
    )
    lines.append("")

    lines.append("## 3. Dog `pca_fixed n_components=2`, k=4 Cluster Sizes")
    lines.append("")
    lines.append(_markdown_table(dog_sizes[["cluster", "count", "ratio_pct"]]))
    lines.append("")

    lines.append("## 4. Dog `pca_fixed n_components=2`, k=4 Cluster Profiles")
    lines.append("")
    lines.append(
        _markdown_table(
            dog_profiles[
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
                ]
            ]
        )
    )
    lines.append("")

    lines.append("## 5. Dog Baseline k=6 vs PCA n_components=2 k=4")
    lines.append("")
    baseline_focus = silhouette_df[
        (silhouette_df["species"] == "Dog")
        & (silhouette_df["method"] == "baseline")
        & (silhouette_df["k"] == FOCUS_BASELINE_DOG_K)
    ]
    if not baseline_focus.empty:
        base_row = baseline_focus.iloc[0]
        lines.append("### Silhouette comparison")
        lines.append("")
        lines.append(
            f"- Baseline k=6: PCA-space={base_row['silhouette_pca_space']:.4f}, "
            f"full-space={base_row['silhouette_full_space']:.4f}"
        )
        if not focus.empty:
            lines.append(
                f"- PCA n=2 k=4: PCA-space={focus.iloc[0]['silhouette_pca_space']:.4f}, "
                f"full-space={focus.iloc[0]['silhouette_full_space']:.4f}"
            )
        lines.append("")

    lines.append("### Baseline k=6 profiles")
    lines.append("")
    lines.append(
        _markdown_table(
            baseline_profiles[
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
                ]
            ]
        )
    )
    lines.append("")

    lines.append("### Interpretation notes")
    lines.append("")
    if not focus.empty and not baseline_focus.empty:
        pca_full = float(focus.iloc[0]["silhouette_full_space"])
        base_full = float(baseline_focus.iloc[0]["silhouette_full_space"])
        if pca_full < base_full:
            lines.append(
                "- In **full encoded feature space**, baseline k=6 has a higher silhouette "
                f"({base_full:.4f}) than PCA n=2 k=4 ({pca_full:.4f}). "
                "The 0.7167 score is not replicated outside the 2D PCA projection."
            )
        else:
            lines.append(
                "- Full-space silhouette for PCA n=2 k=4 remains competitive, but check "
                "whether the gain is driven by a small number of dominant encoded features."
            )

    dog_duration_spread = dog_profiles["duration_mean_days"].max() - dog_profiles["duration_mean_days"].min()
    base_duration_spread = baseline_profiles["duration_mean_days"].max() - baseline_profiles["duration_mean_days"].min()
    lines.append(
        f"- Duration mean spread across clusters: PCA k=4 = {dog_duration_spread:.2f} days, "
        f"baseline k=6 = {base_duration_spread:.2f} days."
    )
    lines.append(
        "- Compare whether cluster profiles separate intake/neuter/breed patterns clearly, "
        "not only whether silhouette is high in PCA space."
    )
    lines.append("")

    if not loadings_df.empty:
        lines.append("## 6. Dog PCA (n=2) Top Encoded-Feature Loadings")
        lines.append("")
        lines.append(
            "Large loadings on a few one-hot columns suggest clusters may be driven by "
            "specific categorical levels rather than broad multi-feature structure."
        )
        lines.append("")
        for component in ["PC1", "PC2"]:
            subset = loadings_df[loadings_df["component"] == component].head(8)
            lines.append(f"### {component}")
            lines.append("")
            lines.append(_markdown_table(subset[["feature", "loading", "abs_loading"]]))
            lines.append("")

    lines.append("## Conclusion For Report Writing")
    lines.append("")
    lines.append(
        "- Treat **PCA-space silhouette** and **full-space silhouette** separately in the report."
    )
    lines.append(
        "- Do not replace the current production clustering based only on PCA-space silhouette "
        "from 2 components unless full-space validation and profile interpretability also improve."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
