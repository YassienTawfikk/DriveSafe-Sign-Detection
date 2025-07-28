import kagglehub
from pathlib import Path
import shutil
import pandas as pd


def download_dataset(dest_folder=Path("data/raw")):
    # Download and get the dataset path
    dataset_path = Path(kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"))
    dataset_subfolder = dataset_path / "GTSRB"

    # Validate expected subfolder
    if not dataset_subfolder.exists():
        raise FileNotFoundError(f"Expected 'GTSRB' folder not found in {dataset_path}")

    # Ensure destination folder exists
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Copy files
    for file in dataset_subfolder.iterdir():
        if file.is_file():
            shutil.copy2(file, dest_folder / file.name)

    return dest_folder
