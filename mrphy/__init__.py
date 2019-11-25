from math import pi as π, inf  # noqa: F401
γH = 4257.6  # Hz/Gauss, gyro frequency of water proton
dt0 = 4e-6   # Sec, default dwell time
T1G = 1.47   # Sec, T1 for gray matter
T2G = 0.07   # Sec, T2 for gray matter

from mrphy import (utils, beffective, sims, slowsims, mobjs)  # noqa: E402

__all__ = ['γH', 'utils', 'beffective', 'sims', 'slowsims', 'mobjs']

"""
*General Comments*:
- `N`:   batch size
- `nM`:  nSpins
- `Nd`:  A int-tuple for array size, e.g.: Nd = (nx, (ny, (nz)))
- `nT`:  the number of time points
- `xy`:  basically means that dimension has length of 2
- `xyz`: means that dimension has length of 3
"""
