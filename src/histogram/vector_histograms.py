import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

class VectorHistograms:
    """Histogram of L2 differences between multivariate real/predicted vectors.

    Expected input shape: (n_windows, window_size, n_signals).
    At each window/time position, the last axis is treated as one N-dimensional
    vector. If normalize_vectors=True, each real and predicted vector is
    independently divided by its own L2 norm before their difference is taken.
    Zero vectors remain zero during normalization.
    """

    def __init__(
        self,
        ground_truth,
        prediction,
        signal_names=None,
        normalize_vectors=False,
    ):
        self.ground_truth = np.asarray(ground_truth, dtype=np.float64)
        self.prediction = np.asarray(prediction, dtype=np.float64)
        self.signal_names = (
            None if signal_names is None else list(signal_names)
        )
        self.normalize_vectors = bool(normalize_vectors)

        self._validate_inputs()
        self.vector_differences = self._calculate_vector_differences()
        self.distances = np.linalg.norm(self.vector_differences, axis=-1)
        self.flat_distances = self.distances.ravel()

    def _validate_inputs(self):
        if self.ground_truth.ndim != 3:
            raise ValueError(
                "ground_truth must have shape (n_windows, window_size, n_signals)."
            )
        if self.prediction.shape != self.ground_truth.shape:
            raise ValueError("ground_truth and prediction must have the same shape.")
        if not np.isfinite(self.ground_truth).all():
            raise ValueError("ground_truth contains non-finite values.")
        if not np.isfinite(self.prediction).all():
            raise ValueError("prediction contains non-finite values.")
        if self.signal_names is not None:
            if len(self.signal_names) != self.ground_truth.shape[-1]:
                raise ValueError("signal_names must contain one name per signal.")
            if len(set(self.signal_names)) != len(self.signal_names):
                raise ValueError("signal_names must be unique.")

    @staticmethod
    def _normalize_vectors(vectors):
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        # A zero vector remains zero; nonzero vectors become unit vectors.
        return np.divide(
            vectors,
            norms,
            out=np.zeros_like(vectors),
            where=norms > 0,
        )

    def _calculate_vector_differences(self):
        real_vectors = self.ground_truth
        predicted_vectors = self.prediction

        if self.normalize_vectors:
            real_vectors = self._normalize_vectors(real_vectors)
            predicted_vectors = self._normalize_vectors(predicted_vectors)

        return real_vectors - predicted_vectors

    def statistics(self):
        """Return descriptive statistics of the vector-difference magnitudes."""
        return pd.Series(self.flat_distances, name="vector_difference_norm").describe()

    def plot(
        self,
        bins=100,
        figsize=(10, 5),
        show=True,
        save_path=None,
        log_scale=False,
        normalize_histogram=False,
        x_range=None,
        y_range=None,
        title=None,
        font_size=12,
    ):
        """Plot the histogram of vector-difference magnitudes.

        normalize_histogram controls relative frequencies and is independent
        of normalize_vectors, which controls vector normalization.
        """
        if bins < 1:
            raise ValueError("bins must be a positive integer.")
        if self.flat_distances.size == 0:
            raise ValueError("There are no vector distances to plot.")

        values = self.flat_distances
        if x_range is not None:
            xmin, xmax = x_range
            if xmin < 0 or xmax <= xmin:
                raise ValueError("x_range must satisfy 0 <= min < max.")
            values_for_range = values[(values >= xmin) & (values <= xmax)]
            if values_for_range.size == 0:
                raise ValueError("No distances fall inside x_range.")
            hist_range = (xmin, xmax)
        else:
            hist_range = (0.0, float(values.max()) if values.max() > 0 else 1.0)

        weights = None
        if normalize_histogram:
            weights = np.full(values.shape, 1.0 / values.size)

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(values, bins=bins, range=hist_range, weights=weights, edgecolor="black")
        if log_scale:
            ax.set_yscale("log")
        if y_range is not None:
            ax.set_ylim(*y_range)

        ax.set_xlabel("Vector Difference Magnitude (L2)", fontsize=font_size)
        ax.set_ylabel(
            "Relative Frequency" if normalize_histogram else "Frequency",
            fontsize=font_size,
        )
        if title is None:
            mode = "Normalized Vectors" if self.normalize_vectors else "Original Scale"
            title = f"Vector Difference Histogram — {mode}"
        ax.set_title(title, fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.grid(True, alpha=0.25)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax
