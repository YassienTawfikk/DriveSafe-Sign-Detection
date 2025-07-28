from src.paths import data_dir, figures_dir

for path in [data_dir, figures_dir]:
    path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory ready: {path}")
