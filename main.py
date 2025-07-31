import pandas as pd
from pathlib import Path
from src.__01__data_setup import download_dataset, save_class_distribution


def main():
    # Step 1: Prepare data
    print("🔽 Downloading GTSRB from KaggleHub...")
    download_dataset()

    print("🔽 Plotting class distribution...")
    save_class_distribution()


if __name__ == "__main__":
    main()
