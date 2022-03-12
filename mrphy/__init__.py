r"""A MRI excitation physics module.

About MRphy.py:
===============

``MRphy.py`` provides the following constant and submodules:

Constant:

    - ``γH``: water proton gyro frequency, "4257.6 Hz/Gauss".

Submodules:

    - :mod:`~mrphy.utils`
    - :mod:`~mrphy.beffective`
    - :mod:`~mrphy.sims`
    - :mod:`~mrphy.slowsims`
    - :mod:`~mrphy.mobjs`

General Comments:
=================

Variable naming convention:
---------------------------

A trailing ``_`` in a variable/attribute name indicates compactness, i.e.
instead of size ``(N, *Nd, ...)``, the variable/attributes size
``(N, nM, ...)``.
For instance:
A field map variable ``Δf`` may be of size ``(N, nx, ny, nz)``, while its
compact countarpart ``Δf_`` has a size of ``(N, nM)``.

Special keywords used in documentations:
----------------------------------------

- ``N``:   batch size
- ``nM``:  the number of spins
- ``Nd``:  A int-tuple for array size, e.g.: ``Nd = (nx, (ny, (nz)))``. \
  In python convention, ``*`` unpacks a tuple. \
  Therefore, ``(N, *Nd) === (N, nx, ny, nz)``.
- ``nT``:  the number of time points
- ``xy``:  the dimension has length of ``2``
- ``xyz``: the dimension has length of ``3``
- ``⊻``: **Either or**. \
  When used in enumerating function keywords arguments, it means the function \
  accepts at most one of the keywords in a set as an input, e.g., \
  ``Δf ⊻ Δf_`` means accepting either ``Δf`` or ``Δf_``. \
  When used in specifying variable dimensions, it means the dimension tuple \
  can be one of the size tuple set, e.g. ``(N, nM ⊻ *Nd, xyz)`` means \
  accepting dimension either ``(N, nM, xyz)`` or ``(N, *Nd, xyz)``.
"""
import warnings, ctypes

from math import pi as π, inf  # noqa: F401
import torch
from torch import tensor
γH = tensor(4257.6, dtype=torch.double)  # Hz/Gauss, water proton gyro freq.
T1G = tensor(1.47, dtype=torch.double)   # Sec, T1 for gray matter
T2G = tensor(0.07, dtype=torch.double)   # Sec, T2 for gray matter

dt0 = tensor(4e-6, dtype=torch.double)   # Sec, default dwell time
gmax0 = tensor(5, dtype=torch.double)    # Gauss/cm
smax0 = tensor(12e3, dtype=torch.double)  # Gauss/cm/Sec
rfmax0 = tensor(0.25, dtype=torch.double)  # Gauss

_slice = slice(None)


def cuda_is_available() -> bool:
    r"""Returns `True` if cuda is available"""
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for name in libnames:
        try:
            ctypes.CDLL(name)
        except OSError:
            continue
        else:
            return True
    else:
        return False
    return False


__CUDA_IS_AVAILABLE__ = cuda_is_available()

try:
    import cupy
    __CUPY_IS_AVAILABLE__ = True
except ImportError:
    if __CUDA_IS_AVAILABLE__:
        warnings.warn('Unable to import `cupy` while CUDA is available',
                      ImportWarning)
    __CUPY_IS_AVAILABLE__ = False


from mrphy import (utils, beffective, sims, slowsims, mobjs)  # noqa: E402
from mrphy.version import __version__

__all__ = ['γH', 'utils', 'beffective', 'sims', 'slowsims', 'mobjs']
