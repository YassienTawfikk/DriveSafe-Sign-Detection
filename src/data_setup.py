from pathlib import Path
import kagglehub
import shutil


def download_dataset(dest_folder=Path("data/raw")):
    # Skip download if data already exists
    if dest_folder.exists() and any(dest_folder.iterdir()):
        print(f"Dataset already exists in {dest_folder}, skipping download.")
        return dest_folder

    # Download dataset
    dataset_path = Path(kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"))

    # Debug: show structure
    print(f"Dataset downloaded to: {dataset_path}")
    print("Files in dataset path:", [f.name for f in dataset_path.iterdir()])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Downloaded dataset path not found: {dataset_path}")

    # Create destination if not exist
    dest_folder.mkdir(parents=True, exist_ok=True)

    for item in dataset_path.iterdir():
        target = dest_folder / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target)

    print(f"Dataset copied to: {dest_folder}")
    return dest_folder
