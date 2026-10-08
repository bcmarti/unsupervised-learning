import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ResidualHistograms:
    """Calculate and visualize residual histograms for multiple signals."""

    def __init__(self, ground_truth, prediction, signal_names):
        self.ground_truth = np.asarray(ground_truth, dtype=np.float32)
        self.prediction = np.asarray(prediction, dtype=np.float32)
        self.signal_names = list(signal_names)

        self._validate_inputs()
        self.residuals = self._calculate_residuals()

    def _validate_inputs(self):
        """Validate input arrays and signal names."""
        if self.ground_truth.ndim != 3:
            raise ValueError(
                "ground_truth must have shape "
                "(n_windows, window_size, n_signals)."
            )

        if self.prediction.ndim != 3:
            raise ValueError(
                "prediction must have shape "
                "(n_windows, window_size, n_signals)."
            )

        if self.ground_truth.shape != self.prediction.shape:
            raise ValueError(
                "ground_truth and prediction must have the same shape."
                f"Received: {self.ground_truth.shape} and "
                f"{self.prediction.shape}."
            )

        if len(self.signal_names) != self.ground_truth.shape[2]:
            raise ValueError(
                "The number of signal names must match the number "
                "of signals in the input arrays."
            )

        if len(set(self.signal_names)) != len(self.signal_names):
            raise ValueError("signal_names must not contain duplicates.")

        if not np.isfinite(self.ground_truth).all():
            raise ValueError("ground_truth contains NaN or Inf values.")

        if not np.isfinite(self.prediction).all():
            raise ValueError("prediction contains NaN or Inf values.")

    def _calculate_residuals(self):
        """Calculate pointwise residuals as ground truth minus prediction."""
        return (self.ground_truth - self.prediction).astype(np.float32)

    def _get_signal_index(self, signal):
        """Return the array index corresponding to a signal."""
        if isinstance(signal, (int, np.integer)):
            if signal < 0 or signal >= len(self.signal_names):
                raise IndexError(f"Signal index out of range: {signal}")
            return int(signal)

        if signal not in self.signal_names:
            raise KeyError(f"Unknown signal: {signal}")

        return self.signal_names.index(signal)

    def _sanitize_filename(self, signal):
        """Return a filesystem-safe filename based on a signal name."""
        sanitized = str(signal)

        for char in '<>:"/\\\\|?*':
            sanitized = sanitized.replace(char, "_")

        return sanitized

    def get_signal_names(self):
        """Return a copy of the signal names."""
        return self.signal_names.copy()

    def get_signal_residuals(self, signal):
        """Return flattened residuals for a single signal."""
        index = self._get_signal_index(signal)

        return self.residuals[:, :, index].ravel().copy()

    def statistics(self, signal):
        """Return descriptive statistics for a single signal."""
        residuals = self.get_signal_residuals(signal)

        return pd.Series(residuals, name=str(signal)).describe()

    def all_statistics(self):
        """Return descriptive statistics for all signals."""
        rows = []

        for signal in self.signal_names:
            stats = self.statistics(signal)

            rows.append(
                {
                    "signal": signal,
                    "count": stats["count"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "25%": stats["25%"],
                    "50%": stats["50%"],
                    "75%": stats["75%"],
                    "max": stats["max"],
                }
            )

        return pd.DataFrame(rows).set_index("signal")

    def plot_signal(
        self,
        signal,
        bins=50,
        figsize=(10, 5),
        show=True,
        save_path=None,
    ):
        """Plot the residual histogram for a single signal.

        Parameters
        ----------
        signal : str or int
            Signal name or signal index.
        bins : int, default=50
            Number of histogram bins.
        figsize : tuple, default=(10, 5)
            Figure size passed to Matplotlib.
        show : bool, default=True
            Whether to display the figure.
        save_path : str or pathlib.Path, optional
            Directory where the histogram will be saved.
        """
        residuals = self.get_signal_residuals(signal)

        plt.figure(figsize=figsize)
        plt.hist(residuals, bins=bins)
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Residual (ground truth - prediction)")
        plt.ylabel("Frequency")
        plt.title(f"Residual histogram — {signal}")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            filename = f"{self._sanitize_filename(signal)}.png"
            plt.savefig(save_path / filename, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_all(
        self,
        bins=50,
        figsize=(10, 5),
        show=True,
        save_path=None,
    ):
        """Plot residual histograms for all signals."""
        for signal in self.signal_names:
            self.plot_signal(
                signal=signal,
                bins=bins,
                figsize=figsize,
                show=show,
                save_path=save_path,
            )