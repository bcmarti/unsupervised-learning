from pathlib import Path
import re

import numpy as np
import pandas as pd


class UnlabeledDataExtractor:
    """Remove temporal sections from unlabeled data that overlap with 3W."""

    def __init__(
        self,
        unlabeled_dir: str | Path,
        dataset_3w_dir: str | Path,
        output_dir: str | Path,
        safety_margin: str | pd.Timedelta = "1min",
        continuity: str | pd.Timedelta = "1s",
    ):
        self.unlabeled_dir = Path(unlabeled_dir)
        self.dataset_3w_dir = Path(dataset_3w_dir)
        self.output_dir = Path(output_dir)

        self.safety_margin = pd.Timedelta(safety_margin)
        self.continuity = pd.Timedelta(continuity)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache the exclusion intervals for each well.
        self._interval_cache = {}

    @staticmethod
    def get_well_id(filename: str) -> str:
        """Extract the well ID from a filename."""
        match = re.match(r"^WELL[_-]?0*(\d+)", filename, re.IGNORECASE)

        if not match:
            raise ValueError(f"Could not identify well ID: {filename}")

        return match.group(1).zfill(5)

    def find_3w_files(self, well_id: str) -> list[Path]:
        """Find all 3W Parquet files belonging to a given well."""
        pattern = f"WELL-{well_id}_*.parquet"

        files = []

        for class_dir in self.dataset_3w_dir.iterdir():
            if class_dir.is_dir() and class_dir.name.isdigit():
                files.extend(class_dir.glob(pattern))

        return sorted(files)

    @staticmethod
    def load_timestamps(file_path: Path) -> pd.DatetimeIndex:
        """Load and validate timestamps from a Parquet file."""
        df = pd.read_parquet(file_path)

        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif "timestamp" in df.columns:
            timestamps = pd.DatetimeIndex(df["timestamp"])
        else:
            raise ValueError(
                f"No DatetimeIndex or 'timestamp' column found in {file_path}"
            )

        if timestamps.hasnans:
            raise ValueError(f"NaT timestamps found in {file_path}")

        return timestamps

    def get_3w_timestamps(self, well_id: str) -> pd.DatetimeIndex:
        """Collect timestamps from every 3W file belonging to a well."""
        files = self.find_3w_files(well_id)

        if not files:
            raise FileNotFoundError(
                f"No 3W files found for well {well_id}"
            )

        timestamps = []

        for file_path in files:
            timestamps.append(self.load_timestamps(file_path))

        return pd.DatetimeIndex(
            np.concatenate([ts.to_numpy() for ts in timestamps])
        ).drop_duplicates().sort_values()

    def build_exclusion_intervals(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Build exclusion intervals from continuous 3W timestamp sections.

        A new section starts whenever consecutive timestamps differ
        from the configured continuity interval.

        Each section is expanded by the safety margin on both sides.
        """
        if len(timestamps) == 0:
            raise ValueError("Cannot build intervals from empty timestamps.")

        timestamps = timestamps.drop_duplicates().sort_values()

        # A new section starts whenever the difference is not exactly 1 second.
        differences = timestamps.to_series().diff()

        section_starts = np.r_[
            0,
            np.flatnonzero(
                differences.iloc[1:].to_numpy() != self.continuity
            ) + 1,
        ]

        section_ends = np.r_[section_starts[1:] - 1, len(timestamps) - 1]

        intervals = []

        for start_idx, end_idx in zip(section_starts, section_ends):
            start = timestamps[start_idx] - self.safety_margin
            end = timestamps[end_idx] + self.safety_margin

            intervals.append((start, end))

        return self.merge_intervals(intervals)

    @staticmethod
    def merge_intervals(
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Merge overlapping or adjacent exclusion intervals."""
        if not intervals:
            return []

        intervals = sorted(intervals)
        merged = [intervals[0]]

        for start, end in intervals[1:]:
            previous_start, previous_end = merged[-1]

            if start <= previous_end:
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))

        return merged

    @staticmethod
    def find_interval_matches(
        timestamps: pd.DatetimeIndex,
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> np.ndarray:
        """
        Return a boolean mask indicating timestamps inside exclusion intervals.

        Interval boundaries are inclusive.
        """
        if not intervals or len(timestamps) == 0:
            return np.zeros(len(timestamps), dtype=bool)

        starts = pd.DatetimeIndex([start for start, _ in intervals])
        ends = pd.DatetimeIndex([end for _, end in intervals])

        # Find the last interval whose start is <= each timestamp.
        positions = starts.searchsorted(timestamps, side="right") - 1

        valid_positions = positions >= 0
        matches = np.zeros(len(timestamps), dtype=bool)

        matches[valid_positions] = (
            timestamps[valid_positions]
            <= ends.take(positions[valid_positions])
        )

        return matches

    def get_exclusion_intervals(
        self,
        well_id: str,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Get cached exclusion intervals for a well."""
        if well_id not in self._interval_cache:
            timestamps = self.get_3w_timestamps(well_id)

            self._interval_cache[well_id] = (
                self.build_exclusion_intervals(timestamps)
            )

        return self._interval_cache[well_id]

    def extract_file(self, input_path: Path) -> dict:
        """Extract samples outside the 3W exclusion intervals."""
        well_id = self.get_well_id(input_path.name)

        intervals = self.get_exclusion_intervals(well_id)

        df = pd.read_parquet(input_path)

        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" not in df.columns:
                raise ValueError(
                    f"No DatetimeIndex or 'timestamp' column in {input_path}"
                )

            df = df.set_index("timestamp")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if df.index.hasnans:
            raise ValueError(f"NaT timestamps found in {input_path}")

        # Identify samples that overlap with the 3W exclusion intervals.
        excluded_mask = self.find_interval_matches(df.index, intervals)

        extracted_df = df.loc[~excluded_mask].copy()

        # Preserve the original filename suffix to avoid overwriting
        # files when a well has multiple unlabeled Parquet files.
        match = re.match(
            r"^WELL[_-]?0*\d+",
            input_path.stem,
            re.IGNORECASE,
        )

        suffix = input_path.stem[match.end():]
        output_path = self.output_dir / f"WELL-{well_id}{suffix}.parquet"

        extracted_df.to_parquet(output_path)

        return {
            "well_id": well_id,
            "input_file": input_path.name,
            "output_file": output_path.name,
            "original_samples": len(df),
            "removed_samples": int(excluded_mask.sum()),
            "remaining_samples": len(extracted_df),
            "removed_percentage": (
                100 * excluded_mask.sum() / len(df)
                if len(df) else 0.0
            ),
        }

    def validate_extraction(self) -> None:
        """
        Validate all generated files.

        Raise ValueError if any output sample belongs to a 3W
        exclusion interval.
        """
        output_files = sorted(self.output_dir.glob("*.parquet"))

        if not output_files:
            raise ValueError("No extracted Parquet files found.")

        total_samples = 0

        for file_path in output_files:
            well_id = self.get_well_id(file_path.name)

            intervals = self.get_exclusion_intervals(well_id)
            timestamps = self.load_timestamps(file_path)

            matches = self.find_interval_matches(timestamps, intervals)

            if matches.any():
                conflicting_timestamps = timestamps[matches]

                raise ValueError(
                    f"Data leakage detected in {file_path.name}: "
                    f"{len(conflicting_timestamps)} timestamps overlap "
                    f"with 3W exclusion intervals. "
                    f"Examples: {conflicting_timestamps[:5].tolist()}"
                )

            total_samples += len(timestamps)

            print(f"[VALIDATED] {file_path.name}")

        print(
            f"\nValidation successful: {len(output_files)} files, "
            f"{total_samples} samples, no temporal overlap detected."
        )

    def print_file_mapping(self) -> None:
        """Print the unlabeled and 3W files associated with each well."""

        unlabeled_files = sorted(self.unlabeled_dir.glob("*.parquet"))

        if not unlabeled_files:
            raise FileNotFoundError(
                f"No Parquet files found in {self.unlabeled_dir}"
            )

        # Group unlabeled files by well.
        wells = {}

        for file_path in unlabeled_files:
            well_id = self.get_well_id(file_path.name)
            wells.setdefault(well_id, []).append(file_path)

        print("\n" + "=" * 80)
        print("UNLABELED / 3W FILE MAPPING")
        print("=" * 80)

        for well_id, files in sorted(wells.items()):
            files_3w = self.find_3w_files(well_id)

            print(f"\nWELL {well_id}")
            print("-" * 80)

            print(f"Unlabeled files ({len(files)}):")

            for file_path in files:
                print(f"  [UNLABELED] {file_path.name}")

            print(f"\n3W files ({len(files_3w)}):")

            if files_3w:
                for file_path in files_3w:
                    relative_path = file_path.relative_to(
                        self.dataset_3w_dir
                    )
                    print(f"  [3W] {relative_path}")
            else:
                print("  [WARNING] No matching 3W files found!")

        print("\n" + "=" * 80)
        print("END OF FILE MAPPING")
        print("=" * 80 + "\n")

    def extract_all(self) -> pd.DataFrame:
        """Extract all unlabeled Parquet files and validate the results."""

        self.print_file_mapping()

        input_files = sorted(self.unlabeled_dir.glob("*.parquet"))

        if not input_files:
            raise FileNotFoundError(
                f"No Parquet files found in {self.unlabeled_dir}"
            )

        results = []

        for input_path in input_files:
            print(f"\nProcessing: {input_path.name}")

            result = self.extract_file(input_path)
            results.append(result)

            print(
                f"  Removed: {result['removed_samples']:,} "
                f"/ {result['original_samples']:,}"
            )
            print(
                f"  Remaining: {result['remaining_samples']:,}"
            )

        report = pd.DataFrame(results)

        print("\nExtraction summary:")
        print(
            report[
                [
                    "well_id",
                    "original_samples",
                    "removed_samples",
                    "remaining_samples",
                    "removed_percentage",
                ]
            ].to_string(index=False)
        )

        print("\nStarting independent validation...")
        self.validate_extraction()

        return report


if __name__ == "__main__":

    # run with 'python utils/extract_unlabeled_without_3w.py 2>&1 | tee extraction.log'
    
    extractor = UnlabeledDataExtractor(
        unlabeled_dir="/home/bruno.martins/dataset",
        dataset_3w_dir="dataset",
        output_dir="unlabeled_dataset_without_3w",
        safety_margin="1min",
        continuity="1s",
    )

    report = extractor.extract_all()

    report.to_csv(
        extractor.output_dir / "extraction_report.csv",
        index=False,
    )