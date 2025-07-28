from src.paths import data_dir, figures_dir
from pathlib import Path
import kagglehub
import shutil
import pandas as pd
import matplotlib.pyplot as plt


def download_dataset(dest_folder=data_dir):
    if dest_folder.exists() and any(dest_folder.iterdir()):
        print(f"✅ Dataset already exists in {dest_folder}, skipping download.")
        return dest_folder

    dataset_path = Path(kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"))

    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

    for item in dataset_path.iterdir():
        target = dest_folder / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)

    print(f"✅ Dataset copied to: {dest_folder}")
    return dest_folder


def save_class_distribution(csv_path=data_dir / "Train.csv", output_dir=figures_dir):
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Train CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    class_counts = df["ClassId"].value_counts().sort_index()

    plt.figure(figsize=(14, 6))
    bars = plt.bar(class_counts.index, class_counts.values, color='skyblue', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 20, str(height),
                 ha='center', va='bottom', fontsize=8)

    plt.xticks(class_counts.index, rotation=90)
    plt.xlabel("Traffic Sign Class ID")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Traffic Sign Classes in Training Set", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = output_dir / "class_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Class distribution plot saved to: {save_path}")
    plt.close()
