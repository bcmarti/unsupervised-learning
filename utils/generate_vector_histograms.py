"""Generate comparable vector-difference histograms from multiple NPZ files."""

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from histogram.vector_histograms import VectorHistograms


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NPZ_PATHS = [
    # First item should be normal operation
    Path(".local_docs/resultado_normal.npz"),
    Path(".local_docs/resultado_anomalia.npz"),
    Path(".local_docs/resultado_transiente.npz"),
]

OUTPUT_ROOT = Path("generated_histograms")

# Fit one shared StandardScaler using the ground-truth samples from ALL files.
# Predictions are transformed with that same scaler.
USE_STANDARD_SCALER = True

# Independently normalize each real/predicted N-dimensional vector by its L2 norm.
NORMALIZE_VECTORS = False

LOG_SCALE = False
NORMALIZE_HISTOGRAM = True
FONT_SIZE = 18

BINS = 100
FIGSIZE = (12, 6)
X_PERCENTILE = 100
X_RANGE = None  # Example: (0.0, 2.0)


SIGNAL_NAMES = [
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
    "P-MON-CKP",
    "P-PDG",
    "P-TPT",
    "T-JUS-CKP",
    "T-TPT",
]


# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------

def load_arrays(npz_path):
    """Load and validate one NPZ file.

    Source format:   (n_windows, n_signals, window_size)
    Required format: (n_windows, window_size, n_signals)
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        if not {"y_pred", "y_true"}.issubset(data.files):
            raise ValueError(
                f"{npz_path} must contain 'y_pred' and 'y_true'."
            )
        y_pred = np.asarray(data["y_pred"], dtype=np.float64)
        y_true = np.asarray(data["y_true"], dtype=np.float64)

    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError(
            f"{npz_path}: y_true and y_pred must both be 3D arrays."
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"{npz_path}: y_true and y_pred must have the same shape. "
            f"Received {y_true.shape} and {y_pred.shape}."
        )

    if y_true.shape[1] != len(SIGNAL_NAMES):
        raise ValueError(
            f"{npz_path}: expected {len(SIGNAL_NAMES)} signals, "
            f"but found {y_true.shape[1]}."
        )

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(f"{npz_path}: arrays contain non-finite values.")

    # (n_windows, n_signals, window_size) -> (n_windows, window_size, n_signals)
    y_true = np.transpose(y_true, (0, 2, 1))
    y_pred = np.transpose(y_pred, (0, 2, 1))

    return y_true, y_pred


def fit_shared_scaler():
    """Fit one StandardScaler using only the normal scenario (first NPZ file).

    The fitted scaler is reused for ground truth and predictions
    from every scenario.
    """
    if not USE_STANDARD_SCALER:
        return None

    # Using normal operation for fit
    normal_npz_path = NPZ_PATHS[0]
    y_true_normal, _ = load_arrays(normal_npz_path)

    scaler = StandardScaler()
    scaler.fit(y_true_normal.reshape(-1, y_true_normal.shape[-1]))

    print(f"Fitted scaler using normal data only: {normal_npz_path}")

    return scaler


def load_analyzer(npz_path, scaler=None):
    """Load one file and create its VectorHistograms analyzer."""
    y_true, y_pred = load_arrays(npz_path)

    if scaler is not None:
        original_shape = y_true.shape
        n_signals = original_shape[-1]

        y_true = scaler.transform(
            y_true.reshape(-1, n_signals)
        ).reshape(original_shape)

        y_pred = scaler.transform(
            y_pred.reshape(-1, n_signals)
        ).reshape(original_shape)

    return VectorHistograms(
        ground_truth=y_true,
        prediction=y_pred,
        signal_names=SIGNAL_NAMES,
        normalize_vectors=NORMALIZE_VECTORS,
    )


# ---------------------------------------------------------------------------
# Shared axis limits
# ---------------------------------------------------------------------------

def calculate_shared_x_range(scaler):
    """Calculate one common nonnegative x-axis range for all files."""
    if X_RANGE is not None:
        xmin, xmax = X_RANGE
        if xmin < 0 or xmax <= xmin:
            raise ValueError("X_RANGE must satisfy 0 <= min < max.")
        return float(xmin), float(xmax)

    if not 0 < X_PERCENTILE <= 100:
        raise ValueError("X_PERCENTILE must be in the interval (0, 100].")

    shared_maximum = 0.0

    for npz_path in NPZ_PATHS:
        analyzer = load_analyzer(npz_path, scaler)
        limit = float(
            np.percentile(analyzer.flat_distances, X_PERCENTILE)
        )
        shared_maximum = max(shared_maximum, limit)
        print(f"Calculated x-axis limit: {npz_path}")
        del analyzer

    # Avoid a zero-width histogram range when all distances are zero.
    if shared_maximum <= 0 or not np.isfinite(shared_maximum):
        shared_maximum = 1.0

    return 0.0, shared_maximum


def calculate_shared_y_range(scaler, shared_x_range):
    """Calculate a common y-axis range using identical histogram bins."""
    shared_maximum = 0.0
    shared_minimum = np.inf
    bin_edges = np.linspace(*shared_x_range, BINS + 1)

    for npz_path in NPZ_PATHS:
        analyzer = load_analyzer(npz_path, scaler)
        distances = analyzer.flat_distances

        weights = (
            np.full(distances.size, 1.0 / distances.size)
            if NORMALIZE_HISTOGRAM and distances.size > 0
            else None
        )

        frequencies, _ = np.histogram(
            distances,
            bins=bin_edges,
            weights=weights,
        )

        positive_frequencies = frequencies[frequencies > 0]

        if positive_frequencies.size:
            shared_maximum = max(
                shared_maximum,
                float(positive_frequencies.max()),
            )
            shared_minimum = min(
                shared_minimum,
                float(positive_frequencies.min()),
            )

        print(f"Calculated y-axis limit: {npz_path}")
        del analyzer

    if shared_maximum <= 0 or not np.isfinite(shared_maximum):
        shared_maximum = 1.0

    if LOG_SCALE:
        if not np.isfinite(shared_minimum):
            shared_minimum = shared_maximum
        # Matplotlib log axes require strictly positive limits.
        shared_minimum = max(shared_minimum, np.finfo(float).tiny)
    else:
        shared_minimum = 0.0

    # Relative histogram frequencies are proportions in [0, 1].
    if NORMALIZE_HISTOGRAM and not LOG_SCALE:
        shared_minimum, shared_maximum = 0.0, 1.0

    return shared_minimum, shared_maximum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Validate inputs, calculate shared axes, and generate all plots."""
    if not NPZ_PATHS:
        raise ValueError("NPZ_PATHS must contain at least one file.")

    if BINS < 1:
        raise ValueError("BINS must be a positive integer.")

    resolved_paths = [path.resolve() for path in NPZ_PATHS]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("NPZ_PATHS contains duplicate files.")

    stems = [path.stem for path in NPZ_PATHS]
    if len(set(stems)) != len(stems):
        raise ValueError(
            "Input files must have distinct filenames (stems) "
            "to avoid output folder collisions."
        )

    if len(set(SIGNAL_NAMES)) != len(SIGNAL_NAMES):
        raise ValueError("SIGNAL_NAMES must not contain duplicates.")

    # Validate all inputs before fitting or plotting.
    for npz_path in NPZ_PATHS:
        load_arrays(npz_path)

    # One common scaler is essential for comparisons between files.
    scaler = fit_shared_scaler()

    shared_x_range = calculate_shared_x_range(scaler)
    shared_y_range = calculate_shared_y_range(scaler, shared_x_range)

    print(f"\nShared x-axis range: {shared_x_range}")
    print(f"Shared y-axis range: {shared_y_range}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for npz_path in NPZ_PATHS:
        analyzer = load_analyzer(npz_path, scaler)

        analyzer.plot(
            bins=BINS,
            figsize=FIGSIZE,
            show=False,
            save_path=OUTPUT_ROOT / f"{npz_path.stem}.png",
            log_scale=LOG_SCALE,
            normalize_histogram=NORMALIZE_HISTOGRAM,
            x_range=shared_x_range,
            y_range=shared_y_range,
            font_size=FONT_SIZE,
            title=f"Vector Difference Histogram — {npz_path.stem}",
        )

        statistics_path = (
            OUTPUT_ROOT / f"{npz_path.stem}_statistics.csv"
        )
        analyzer.statistics().to_csv(
            statistics_path,
            header=True,
        )

        print(f"\nProcessed: {npz_path}")
        print(f"Histogram: {(OUTPUT_ROOT / f'{npz_path.stem}.png').resolve()}")
        print(f"Statistics: {statistics_path.resolve()}")

        del analyzer

    print("\nAll vector histograms were generated.")


if __name__ == "__main__":
    main()
