import shutil
from pathlib import Path

from ThreeWToolkit.dataset import ParquetDatasetConfig


def main():
    dataset_path = Path("dataset")

    ParquetDatasetConfig(
        path=dataset_path,
        version="2.0.0",
        # force_download=True,
    ).build()

    download_path = dataset_path / "download"

    if download_path.exists():
        shutil.rmtree(download_path)

    print("3W Dataset downloaded successfully.")


if __name__ == "__main__":
    main()