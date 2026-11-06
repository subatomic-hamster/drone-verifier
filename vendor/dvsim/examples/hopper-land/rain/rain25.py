"""Model script of rain25.scenario.toml: the scenario's scene at its default parameters (what `dvsim program` loads and the viewer shows)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import landscene

build = landscene.build_for(Path(__file__).with_name("rain25.scenario.toml"))
