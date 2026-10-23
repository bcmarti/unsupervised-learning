"""Generate residual histograms from an NPZ file."""

from pathlib import Path

import numpy as np

from histogram.residual_histograms import ResidualHistograms


# Configuration
NPZ_PATH = Path("Path/to/.npz")
log_scale = False

BASE_OUTPUT_DIR = Path("histograms") / NPZ_PATH.stem

OUTPUT_DIR = (
    BASE_OUTPUT_DIR.with_name(f"{BASE_OUTPUT_DIR.name}_log")
    if log_scale
    else BASE_OUTPUT_DIR
)

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

def main():
    """Load predictions, reorganize arrays, and generate histograms."""
    if not NPZ_PATH.is_file():
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the arrays from the NPZ file.
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        y_pred = data["y_pred"]
        y_true = data["y_true"]

    print("Original array shapes:")
    print(f"y_true: {y_true.shape}")
    print(f"y_pred: {y_pred.shape}")

    # Reorganize arrays from (n_windows, n_signals, window_size)
    # to (n_windows, window_size, n_signals).
    y_true = np.transpose(y_true, (0, 2, 1))
    y_pred = np.transpose(y_pred, (0, 2, 1))

    print("\nReorganized array shapes:")
    print(f"y_true: {y_true.shape}")
    print(f"y_pred: {y_pred.shape}")

    # Initialize the residual histogram analyzer.
    analyzer = ResidualHistograms(
        ground_truth=y_true,
        prediction=y_pred,
        signal_names=SIGNAL_NAMES,
    )

    # Generate and save one histogram per signal.
    analyzer.plot_all(
        bins=101,
        figsize=(12, 6),
        show=False,
        save_path=OUTPUT_DIR,
        log_scale=log_scale,
        x_percentile=100,
    )

    # Save descriptive statistics for all signals.
    statistics = analyzer.all_statistics()
    statistics.to_csv(OUTPUT_DIR / "residual_statistics.csv")

    print(f"\nGenerated residual histograms.")    
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Statistics saved to: {OUTPUT_DIR / 'residual_statistics.csv'}")


if __name__ == "__main__":
    main()