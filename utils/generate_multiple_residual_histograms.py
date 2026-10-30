"""Generate comparable residual histograms from multiple NPZ files."""

from pathlib import Path

import numpy as np

from histogram.residual_histograms import ResidualHistograms


# Configuration
# Add multiple cases for comparison:
NPZ_PATHS = [
    Path(".local_docs/resultado_normal.npz"),
    Path(".local_docs/resultado_anomalia.npz"),
    Path(".local_docs/resultado_transiente.npz"),
]

OUTPUT_ROOT = Path("histograms")

LOG_SCALE = True
NORMALIZE = True
FONT_SIZE = 12

BINS = 101
FIGSIZE = (12, 6)
X_PERCENTILE = 100
X_RANGE = None  # Example: (-3, 3)

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


def load_analyzer(npz_path):
    """Load one NPZ file and create its residual histogram analyzer."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        if not {"y_pred", "y_true"}.issubset(data.files):
            raise ValueError(
                f"{npz_path} must contain 'y_pred' and 'y_true'."
            )

        y_pred = data["y_pred"]
        y_true = data["y_true"]

    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError(
            f"{npz_path}: y_true and y_pred must both be 3D arrays."
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"{npz_path}: y_true and y_pred must have the same shape. "
            f"Received {y_true.shape} and {y_pred.shape}."
        )

    # Source format: (n_windows, n_signals, window_size).
    # Required format: (n_windows, window_size, n_signals).
    y_true = np.transpose(y_true, (0, 2, 1))
    y_pred = np.transpose(y_pred, (0, 2, 1))

    return ResidualHistograms(
        ground_truth=y_true,
        prediction=y_pred,
        signal_names=SIGNAL_NAMES,
    )


def calculate_shared_x_ranges():
    """Calculate one symmetric x-axis range per signal across all files."""
    if X_RANGE is not None:
        validated_range = ResidualHistograms._validate_x_range(X_RANGE)
        return {
            signal: validated_range
            for signal in SIGNAL_NAMES
        }

    if not 0 < X_PERCENTILE <= 100:
        raise ValueError("X_PERCENTILE must be in the interval (0, 100].")

    # Initialize the largest limit for each signal.
    shared_limits = {
        signal: 0.0
        for signal in SIGNAL_NAMES
    }

    for npz_path in NPZ_PATHS:
        analyzer = load_analyzer(npz_path)

        for signal in SIGNAL_NAMES:
            residuals = analyzer.get_signal_residuals(signal)

            limit = float(
                np.percentile(np.abs(residuals), X_PERCENTILE)
            )

            shared_limits[signal] = max(
                shared_limits[signal],
                limit,
            )

        print(f"Calculated x-axis limits: {npz_path}")

        # Release the current analyzer before loading the next file.
        del analyzer

    # Handle signals whose residuals are zero in every file.
    return {
        signal: (
            -limit if limit > 0 else -1.0,
            limit if limit > 0 else 1.0,
        )
        for signal, limit in shared_limits.items()
    }


def calculate_shared_y_ranges(shared_x_ranges):
    """Calculate shared y-axis ranges per signal across all files."""
    shared_maxima = {signal: 0.0 for signal in SIGNAL_NAMES}
    shared_minima = {signal: np.inf for signal in SIGNAL_NAMES}

    for npz_path in NPZ_PATHS:
        analyzer = load_analyzer(npz_path)

        for signal in SIGNAL_NAMES:
            residuals = analyzer.get_signal_residuals(signal)

            bin_edges = np.linspace(
                *shared_x_ranges[signal],
                BINS + 1,
            )

            weights = (
                np.full(residuals.size, 1.0 / residuals.size)
                if NORMALIZE and residuals.size > 0
                else None
            )

            frequencies, _ = np.histogram(
                residuals,
                bins=bin_edges,
                weights=weights,
            )

            positive_frequencies = frequencies[frequencies > 0]

            if positive_frequencies.size:
                shared_maxima[signal] = max(
                    shared_maxima[signal],
                    float(positive_frequencies.max()),
                )

                shared_minima[signal] = min(
                    shared_minima[signal],
                    float(positive_frequencies.min()),
                )

        print(f"Calculated y-axis limits: {npz_path}")
        del analyzer

    shared_ranges = {}

    for signal in SIGNAL_NAMES:
        y_max = 1.0 if NORMALIZE else shared_maxima[signal]
        y_min = shared_minima[signal]

        if y_max <= 0 or not np.isfinite(y_max):
            y_max = 1.0

        if not np.isfinite(y_min):
            y_min = y_max

        if not LOG_SCALE:
            y_min = 0.0

        shared_ranges[signal] = (y_min, y_max)

    return shared_ranges


def main():
    """Validate inputs, calculate shared ranges, and generate plots."""
    if not NPZ_PATHS:
        raise ValueError("NPZ_PATHS must contain at least one file.")

    resolved_paths = [path.resolve() for path in NPZ_PATHS]

    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("NPZ_PATHS contains duplicate files.")

    # Prevent different input files from writing to the same output folder.
    stems = [path.stem for path in NPZ_PATHS]

    if len(set(stems)) != len(stems):
        raise ValueError(
            "Input files must have distinct filenames (stems) "
            "to avoid output folder collisions."
        )

    if len(set(SIGNAL_NAMES)) != len(SIGNAL_NAMES):
        raise ValueError("SIGNAL_NAMES must not contain duplicates.")

    # Check every file before starting the plotting process.
    # The analyzer validates the number of signals against SIGNAL_NAMES.
    for npz_path in NPZ_PATHS:
        analyzer = load_analyzer(npz_path)
        del analyzer

    # Calculate common limits for each corresponding signal.
    shared_x_ranges = calculate_shared_x_ranges()
    shared_y_ranges = calculate_shared_y_ranges(shared_x_ranges)

    print("\nShared x-axis ranges:")
    for signal, limits in shared_x_ranges.items():
        print(f"{signal}: {limits}")

    print("\nShared y-axis ranges:")
    for signal, limits in shared_y_ranges.items():
        print(f"{signal}: {limits}")

    # Generate plots separately for each input file.
    for npz_path in NPZ_PATHS:
        analyzer = load_analyzer(npz_path)

        base_output_dir = OUTPUT_ROOT / npz_path.stem

        output_dir = (
            base_output_dir.with_name(f"{base_output_dir.name}_log")
            if LOG_SCALE
            else base_output_dir
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer.plot_all(
            bins=BINS,
            figsize=FIGSIZE,
            show=False,
            save_path=output_dir,
            log_scale=LOG_SCALE,
            x_percentile=X_PERCENTILE,
            font_size=FONT_SIZE,
            normalize=NORMALIZE,
            signal_x_ranges=shared_x_ranges,
            signal_y_ranges=shared_y_ranges,
        )

        statistics_path = output_dir / "residual_statistics.csv"
        analyzer.all_statistics().to_csv(statistics_path)

        print(f"\nProcessed: {npz_path}")
        print(f"Output directory: {output_dir.resolve()}")
        print(f"Statistics saved to: {statistics_path.resolve()}")

        del analyzer

    print("\nAll residual histograms were generated.")


if __name__ == "__main__":
    main()