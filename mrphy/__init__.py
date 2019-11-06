γH = 4257.6  # Hz/Gauss, gyro frequency of water proton
dt0 = 4e-6   # Sec, default dwell time

from math import pi as π, inf
from mrphy import (utils, sims, mobjs)

__all__ = ['γH', 'utils', 'sims', 'mobjs']
"""
*General Comments*:
- `N`:   batch size
- `nM`:  nSpins
- `Nd`:  A int-tuple for array size, e.g.: Nd = (nx, (ny, (nz)))
- `nT`:  the number of time points
- `xy`:  basically means that dimension has length of 2
- `xyz`: means that dimension has length of 3
"""

