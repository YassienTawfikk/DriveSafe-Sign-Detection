from src.__00__paths import *

for path in [data_dir, figures_dir, model_dir, docs_dir]:
    path.mkdir(parents=True, exist_ok=True)
