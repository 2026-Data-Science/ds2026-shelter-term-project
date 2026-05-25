from __future__ import annotations

import sys

from src.clustering.analysis import (
    DURATION_BINS_FILENAME,
    DURATION_BOXPLOT_FILENAME,
    OUTPUT_DIR,
    PCA_SCATTER_FILENAME,
    REPORT_PATH,
    run_long_stay_workflow,
)
from src.clustering.preprocessing import ENRICHED_TRAIN_PATH, PROJECT_ROOT


# Force UTF-8 on Windows consoles that default to cp949.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main() -> None:
    """Run only the long-stay risk clustering workflow."""
    print("=" * 60)
    print("Long-stay risk clustering — intake join + K-Means profiling")
    print("=" * 60)

    results, enriched, clustered = run_long_stay_workflow()

    print(f"[1/4] Enriched train CSV written: {ENRICHED_TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[2/4] Rows usable for clustering/profile: {len(clustered):,} / {len(enriched):,}")
    for species, result in results.items():
        print(
            f"[3/4] {species}: selected K={result.selected_k}, "
            f"silhouette={result.silhouette_sample:.4f}, "
            f"encoded={result.encoded_shape}, pca={result.pca_shape}"
        )

    print(f"[4/4] Report written: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"      PCA scatter: {(OUTPUT_DIR / PCA_SCATTER_FILENAME).relative_to(PROJECT_ROOT)}")
    print(f"      Duration boxplot: {(OUTPUT_DIR / DURATION_BOXPLOT_FILENAME).relative_to(PROJECT_ROOT)}")
    print(f"      Duration bins: {(OUTPUT_DIR / DURATION_BINS_FILENAME).relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    print("Long-stay risk clustering completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
