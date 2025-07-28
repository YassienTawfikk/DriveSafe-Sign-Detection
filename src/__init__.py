from pathlib import Path

# The base directory is the project root (parent of 'src')
base_dir = Path(__file__).resolve().parents[1]

# Target directories to create
paths = [
    base_dir / "data" / "raw",
]

for path in paths:
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")
