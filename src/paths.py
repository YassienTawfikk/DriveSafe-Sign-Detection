from pathlib import Path


def get_base_dir():
    """Return the project root whether from .py or .ipynb"""
    here = Path().resolve()
    if (here / "src").exists():
        return here
    elif (here.name == "src") or (here / "../data").exists():
        return here.parent
    else:
        return here.parents[1]


base_dir = get_base_dir()
data_dir = base_dir / "data"
outputs_dir = base_dir / "outputs"
figures_dir = outputs_dir / "figures"
model_dir = outputs_dir / "model"
