import pandas as pd
from src.data_setup import download_dataset


def main():
    # Step 1: Prepare data
    print("🔽 Downloading GTSRB from KaggleHub...")
    download_dataset()


if __name__ == "__main__":
    main()
