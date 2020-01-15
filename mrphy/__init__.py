from math import pi as π, inf  # noqa: F401
import torch
from torch import tensor
γH = tensor(4257.6, dtype=torch.double)  # Hz/Gauss, water proton gyro freq.
T1G = tensor(1.47, dtype=torch.double)   # Sec, T1 for gray matter
T2G = tensor(0.07, dtype=torch.double)   # Sec, T2 for gray matter

dt0 = tensor(4e-6, dtype=torch.double)   # Sec, default dwell time
gmax0 = tensor(5, dtype=torch.double)    # Gauss/cm
smax0 = tensor(12e3, dtype=torch.double) # Gauss/cm/Sec
rfmax0 = tensor(0.25, dtype=torch.double) # Gauss

_slice = slice(None)

from mrphy import (utils, beffective, sims, slowsims, mobjs)  # noqa: E402

__all__ = ['γH', 'utils', 'beffective', 'sims', 'slowsims', 'mobjs']

"""
*General Comments*:
- `N`:   batch size
- `nM`:  the number of spins
- `Nd`:  A int-tuple for array size, e.g.: Nd = (nx, (ny, (nz)))
- `nT`:  the number of time points
- `xy`:  basically means that dimension has length of 2
- `xyz`: means that dimension has length of 3

A trailing `_` in a variable/attribute name indicates compactness, i.e. instead
of size (N, *Nd, ...), the variable/attributes size (N, nM, ...).
"""
