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
                "ground_truth and prediction must have the same shape. "
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
        """Calculate residuals as ground truth minus prediction."""
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

    @staticmethod
    def _sanitize_filename(signal):
        """Return a filesystem-safe filename."""
        sanitized = str(signal)

        for char in '<>:"/\\|?*':
            sanitized = sanitized.replace(char, "_")

        return sanitized

    @staticmethod
    def _validate_x_range(x_range):
        """Validate an explicit x-axis interval."""
        if x_range is None:
            return None

        if len(x_range) != 2:
            raise ValueError("x_range must contain (x_min, x_max).")

        x_min, x_max = x_range

        if not np.isfinite([x_min, x_max]).all() or x_min >= x_max:
            raise ValueError(
                "x_range must contain finite values with x_min < x_max."
            )

        return float(x_min), float(x_max)

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

            rows.append({
                "signal": signal,
                "count": stats["count"],
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "25%": stats["25%"],
                "50%": stats["50%"],
                "75%": stats["75%"],
                "max": stats["max"],
            })

        return pd.DataFrame(rows).set_index("signal")

    def _get_default_x_range(self, signal, x_percentile):
        """Calculate a symmetric x-axis range from absolute residuals."""
        if not 0 < x_percentile <= 100:
            raise ValueError(
                "x_percentile must be in the interval (0, 100]."
            )

        residuals = self.get_signal_residuals(signal)
        limit = float(np.percentile(np.abs(residuals), x_percentile))

        if limit == 0:
            limit = 1.0

        return -limit, limit

    def _get_default_y_range(
        self,
        signal,
        bins,
        x_range,
        normalize=False,
    ):
        """Calculate the y-axis range from histogram frequencies."""
        residuals = self.get_signal_residuals(signal)

        bin_edges = np.linspace(
            x_range[0],
            x_range[1],
            bins + 1,
        )

        weights = (
            np.full(residuals.size, 1.0 / residuals.size)
            if normalize and residuals.size > 0
            else None
        )

        frequencies, _ = np.histogram(
            residuals,
            bins=bin_edges,
            weights=weights,
        )

        y_max = float(frequencies.max())

        if y_max == 0:
            y_max = 1.0

        return 0.0, y_max

    def plot_signal(
        self,
        signal,
        bins=100,
        figsize=(10, 5),
        show=True,
        save_path=None,
        log_scale=True,
        x_percentile=100,
        font_size=12,
        normalize=False,
        x_range=None,
        y_range=None,
    ):
        """Plot a residual histogram for a single signal.

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
            Percentile of absolute residuals used for the default
            symmetric x-axis range. Ignored when x_range is supplied.
        font_size : float, default=12
            Base font size for plot text.
        normalize : bool, default=False
            If True, show relative frequencies instead of counts.
        x_range : tuple, optional
            Explicit x-axis limits, e.g. (-3, 3). Overrides
            x_percentile.
        y_range : tuple, optional
                    Explicit y-axis limits, e.g. (0, 100).
        """
        if not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError("bins must be a positive integer.")

        if not np.isfinite(font_size) or font_size <= 0:
            raise ValueError("font_size must be positive and finite.")

        residuals = self.get_signal_residuals(signal)

        x_range = self._validate_x_range(x_range)

        if x_range is None:
            x_min, x_max = self._get_default_x_range(
                signal, x_percentile
            )
        else:
            x_min, x_max = x_range

        n_outside = np.count_nonzero(
            (residuals < x_min) | (residuals > x_max)
        )

        bin_edges = np.linspace(x_min, x_max, bins + 1)

        # Relative frequency: each bin shows its fraction of observations.
        weights = (
            np.full(residuals.size, 1.0 / residuals.size)
            if normalize and residuals.size > 0
            else None
        )

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            residuals,
            bins=bin_edges,
            weights=weights,
            log=log_scale,
        )

        if y_range is not None:
            ax.set_ylim(*y_range)

        ax.axvline(
            0,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Zero residual",
        )

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(
            "Residual (ground truth - prediction)",
            fontsize=font_size,
        )

        if normalize:
            ylabel = "Relative frequency"
            if log_scale:
                ylabel += " (log scale)"
        else:
            ylabel = "Frequency"
            if log_scale:
                ylabel += " (log scale)"

        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_title(
            f"Residual histogram — {signal}",
            fontsize=font_size,
        )
        ax.tick_params(axis="both", labelsize=font_size)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=font_size)

        if n_outside:
            ax.text(
                0.02,
                0.95,
                f"{n_outside:,} observations outside displayed range",
                transform=ax.transAxes,
                va="top",
                fontsize=font_size * 0.8,
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
        x_percentile=100,
        font_size=12,
        normalize=False,
        x_range=None,
        signal_x_ranges=None,
        y_range=None,
        signal_y_ranges=None,
    ):
        """Plot residual histograms for all signals.

        signal_x_ranges is an optional dictionary mapping each signal
        name to its own (x_min, x_max) interval. It is useful when
        comparing multiple files.
        """
        for signal in self.signal_names:
            current_x_range = (
                signal_x_ranges[signal]
                if signal_x_ranges is not None
                else x_range
            )

            current_y_range = (
                signal_y_ranges[signal]
                if signal_y_ranges is not None
                else y_range
            )

            self.plot_signal(
                signal=signal,
                bins=bins,
                figsize=figsize,
                show=show,
                save_path=save_path,
                log_scale=log_scale,
                x_percentile=x_percentile,
                font_size=font_size,
                normalize=normalize,
                x_range=current_x_range,
                y_range=current_y_range,
            )
