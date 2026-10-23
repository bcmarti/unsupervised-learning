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
        bins=100,
        figsize=(10, 5),
        show=True,
        save_path=None,
        log_scale=True,
        x_percentile=99.5,
    ):
        """Plot a centered residual histogram for a single signal.

        Parameters
        ----------
        signal : str or int
            Signal name or signal index.
        bins : int, default=100
            Number of histogram bins.
        figsize : tuple, default=(10, 5)
            Figure size passed to Matplotlib.
        show : bool, default=True
            Whether to display the figure.
        save_path : str or pathlib.Path, optional
            Directory where the histogram will be saved.
        log_scale : bool, default=True
            Whether to use a logarithmic frequency axis.
        x_percentile : float, default=99.5
            Percentile of absolute residuals used to set symmetric
            x-axis limits. Values outside the limits are omitted
            from the histogram but counted in the annotation.
        """
        residuals = self.get_signal_residuals(signal)

        if not 0 < x_percentile <= 100:
            raise ValueError("x_percentile must be in the interval (0, 100].")

        # Use a symmetric range so that zero is always centered.
        limit = np.percentile(np.abs(residuals), x_percentile)

        # Handle signals whose residuals are all zero.
        if limit == 0:
            limit = 1.0

        x_min = -limit
        x_max = limit

        # Count observations outside the displayed range.
        n_outside = np.count_nonzero(np.abs(residuals) > limit)

        # Use symmetric bin edges aligned around zero.
        bin_edges = np.linspace(x_min, x_max, bins + 1)

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            residuals,
            bins=bin_edges,
            log=log_scale,
        )

        ax.axvline(
            0,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Zero residual",
        )

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Residual (ground truth - prediction)")
        ax.set_ylabel(
            "Frequency (log scale)" if log_scale else "Frequency"
        )
        ax.set_title(f"Residual histogram — {signal}")
        ax.grid(alpha=0.2)
        ax.legend()

        if n_outside:
            ax.text(
                0.02,
                0.95,
                f"{n_outside:,} observations outside displayed range",
                transform=ax.transAxes,
                va="top",
            )

        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            filename = f"{self._sanitize_filename(signal)}.png"
            fig.savefig(save_path / filename, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    
    def plot_all(
        self,
        bins=100,
        figsize=(10, 5),
        show=True,
        save_path=None,
        log_scale=True,
        x_percentile=99.5,
    ):
        """Plot residual histograms for all signals.

        Parameters
        ----------
        bins : int, default=100
            Number of histogram bins.
        figsize : tuple, default=(10, 5)
            Figure size passed to Matplotlib.
        show : bool, default=True
            Whether to display each figure.
        save_path : str or pathlib.Path, optional
            Directory where the histograms will be saved.
        log_scale : bool, default=True
            Whether to use a logarithmic frequency axis.
        x_percentile : float, default=99.5
            Percentile of absolute residuals used to set symmetric
            x-axis limits for each signal.
        """
        for signal in self.signal_names:
            self.plot_signal(
                signal=signal,
                bins=bins,
                figsize=figsize,
                show=show,
                save_path=save_path,
                log_scale=log_scale,
                x_percentile=x_percentile,
            )
