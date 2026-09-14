"""Import entry for the selective population membrane-memory prototype, not exact fractional LIF."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f_lif_pop_v2.forecasting.layers import PopulationLIF
from f_lif_pop_v2.forecasting.ours import myModel

Model = myModel
