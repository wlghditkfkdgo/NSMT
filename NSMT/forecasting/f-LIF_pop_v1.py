"""Population-selective membrane memory forecasting model (provisional filename).

Implementation: NSMT/f_lif_pop_v1/forecasting/{ours.py,layers.py}.
This entry keeps the originally requested filename. The task directory follows
model_v1's config/model/train/test/utils layout; see its README and PROJECT_LOG.

One logical neuron contains K heterogeneous LIF constituents. At each ordered
patch, the current population queries earlier post-reset membrane vectors.
Retrieved memory adds evidence before spike/reset: v = u_bar + gamma*g*memory.
The first experiment uses shared learned cosine Q/K, a sigmoid gate, subtractive
reset, K spikes, and full BPTT. State and memory reset for each input window.
Direct state retrieval is not an exact fractional differential equation.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f_lif_pop_v1.forecasting.ours import myModel
from f_lif_pop_v1.forecasting.layers import PopulationLIF

Model = myModel
__all__ = ['myModel', 'Model', 'PopulationLIF']
