# Data locations:
from pathlib import Path

data_dir = Path(__file__).parent.parent / 'data'

file_tas      = data_dir / "tas_ipcc.csv"
file_sherwood = data_dir / "sherwood_ecs.json"
file_zelinka  = data_dir / "zelinka_full.json"
