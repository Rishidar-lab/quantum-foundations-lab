"""PHASE 3 / PHASE 4 -- experiment laboratories.

    experiments.teleport_lab -- teleportation run as a measured experiment
    experiments.noise_lab    -- controlled decoherence experiments

Both distinguish, in every result they emit:
    * a STATEVECTOR PROBABILITY  (exact, from the ideal simulator)
    * a FINITE-SHOT FREQUENCY     (an estimate, with a stated confidence interval)

Every emitted JSON record is tagged  "source": "SIMULATED"  -- nothing in this
package fabricates hardware data.
"""
from __future__ import annotations

import pathlib
import sys

_KATAS = str(pathlib.Path(__file__).resolve().parent.parent / "katas")
if _KATAS not in sys.path:
    sys.path.insert(0, _KATAS)
