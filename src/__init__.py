from src.paths import *

for path in [data_dir, figures_dir, model_dir]:
    path.mkdir(parents=True, exist_ok=True)
    print(f"init.py is called")
