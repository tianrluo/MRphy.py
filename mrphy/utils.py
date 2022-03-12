r"""MRphy utilities

Utilities for data indexing, conversions, spin rotation.
"""

from typing import Any, Tuple, Union
from numbers import Number

import torch
import numpy as np
from numpy import ndarray as ndarray_c
from torch import Tensor

from mrphy import γH, dt0, π, __CUPY_IS_AVAILABLE__
if __CUPY_IS_AVAILABLE__:
    import cupy as cp
    from cupy import ndarray as ndarray_g
    ndarrayA = Union[ndarray_c, ndarray_g]
else:
    ndarrayA = ndarray_c


__all__ = ['ctrsub', 'g2k', 'g2s', 'k2g', 'rf_c2r', 'rf_r2c', 'rf2tρθ',
           'rfclamp', 's2g', 's2ts', 'sclamp', 'ts2s', 'tρθ2rf', 'uφrot']


def ctrsub(shape: Any) -> Any:
    r"""Compute center subscript indices of a regular grid

    Usage:
        ``cSub = ctrsub(shape)``
    """
    return shape//2


def g2k(g: Tensor, isTx: bool, dt: Tensor = dt0, *, γ: Tensor = γH) -> Tensor:
    r"""Compute k-space from gradients.

    Usage:
        ``k = g2k(g, isTx, dt, *, γ)``

    Inputs:
        - ``g``: `(N, xyz, nT)`, "Gauss/cm", gradient
        - ``isTx``, if ``true``, compute transmit k-space, `k`, ends at the \
          origin.
    Optionals:
        - ``γ``:  `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Hz/Gauss", gyro ratio
        - ``dt``: `()` ⊻ `(N ⊻ 1,)`, "Sec", dwell time.
    Outputs:
        - ``k``: `(N, xyz, nT)`, "cycle/cm", Tx or Rx k-space.

    See Also:
        :func:`~mrphy.utils.g2s`, :func:`~mrphy.utils.k2g`
    """
    # (N, xyz, nT) compatible
    ndim = g.ndim
    γ, dt = (x.reshape(x.shape+(ndim-x.ndim)*(1,)) for x in (γ, dt))

    k = γ * dt * torch.cumsum(g, dim=2)
    if isTx:
        k -= k[:, :, [-1]]
    return k


def g2s(g: Tensor, dt: Tensor = dt0) -> Tensor:
    r"""Compute slew rates from gradients.

    Usage:
        ``s = g2s(g, dt)``
    Inputs:
        - ``g``: `(N, xyz, nT)`, "Gauss/cm", gradient
    Optionals:
        - ``dt``: `()` ⊻ `(N ⊻ 1,)`, "Sec", dwell time.
    Outputs:
        - ``s``: `(N, xyz, nT)`, "cycle/cm/sec", slew rate

    See Also:
        :func:`~mrphy.utils.g2k`, :func:`~mrphy.utils.s2g`
    """
    dt = dt.reshape(dt.shape+(g.ndim-dt.ndim)*(1,))

    s = torch.cat((g[:, :, [0]], g[:, :, 1:] - g[:, :, :-1]), dim=2)/dt
    return s


def k2g(k: Tensor, isTx: bool, dt: Tensor = dt0, *, γ: Tensor = γH) -> Tensor:
    r"""Compute k-space from gradients

    Usage:
        ``k = k2g(k, isTx, dt, *, γ)``

    Inputs:
        - ``k``: `(N, xyz, nT)`, "cycle/cm", Tx or Rx k-space.
        - ``isTx``, if ``true``, compute transmit k-space, ``k``, must end at \
          the origin.
    Optionals:
        - ``γ``:  `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Hz/Gauss", gyro ratio
        - ``dt``: `()` ⊻ `(N ⊻ 1,)`, "Sec", dwell time.
    Outputs:
        - ``g``: `(N, xyz, nT)`, "Gauss/cm", gradient

    See Also:
        :func:`~mrphy.utils.g2k`
    """
    assert((not isTx) or torch.all(k[:, :, -1] == 0))  # Tx k must end at 0

    ndim = k.ndim
    γ, dt = (x.reshape(x.shape+(ndim-x.ndim)*(1,)) for x in (γ, dt))

    g = torch.cat((k[:, :, [0]], k[:, :, 1:] - k[:, :, :-1]), dim=2)/γ/dt
    return g


def lρθ2rf(lρ: Tensor, θ: Tensor, rfmax: Tensor) -> Tensor:
    r"""Convert tρ ≔ tan(ρ/ρ_max⋅π/2), and θ to real RF

    Usage:
        ``rf = lρθ2rf(lρ, θ, rfmax)``
    Inputs:
        - ``lρ``: `(N, 1, nT, (nCoils))`, logit(ρ/rfmax), [-∞, +∞).
        - ``θ``: `(N, 1, nT, (nCoils))`, RF phase, [-π, π).
        - ``rfmax``: `(N, (nCoils))`, RF pulse, Gauss, x for real, y for imag.
    Outputs:
        - ``rf``: `(N, xy, nT, (nCoils))`, RF pulse, Gauss, x: real, y: imag.

    See Also:
        :func:`~mrphy.utils.rf2lρθ`
    """
    rfmax = rfmax[None] if rfmax.ndim == 0 else rfmax
    rfmax = rfmax[:, None, None, ...]  # -> (N, 1, 1, (nCoils))
    return lρ.sigmoid()*rfmax*torch.cat((θ.cos(), θ.sin()), dim=1)


def rf_c2r(rf: ndarrayA) -> ndarrayA:
    r"""Convert complex RF to real RF

    Usage:
        ``rf = rf_c2r(rf)``
    Inputs:
        - ``rf``: `(N, 1, nT, (nCoils))`, RF pulse, complex
    Outputs:
        - ``rf``: `(N, xy, nT, (nCoils))`, RF pulse, x for real, y for imag.

    See Also:
        :func:`~mrphy.utils.rf_r2c`
    """
    if isinstance(rf, ndarray_c):
        return np.concatenate((np.real(rf), np.imag(rf)), axis=1)
    elif __CUPY_IS_AVAILABLE__:  # ndarray_g, i.e., cupy.ndarray
        return cp.concatenate((cp.real(rf), cp.imag(rf)), axis=1)
    else:
        raise TypeError(f'Unknown type: {type(rf)}')


def rf_r2c(rf: ndarrayA) -> ndarrayA:
    r"""Convert real RF to complex RF

    Usage:
        ``rf = rf_r2c(rf)``
    Inputs:
        - ``rf``: `(N, xy, nT, (nCoils))`, RF pulse, x for real, y for imag.
    Outputs:
        - ``rf``: `(N, 1, nT, (nCoils))`, RF pulse, complex.

    See Also:
        :func:`~mrphy.utils.rf_c2r`
    """
    return rf[:, [0], ...] + 1j*rf[:, [1], ...]


def rf2lρθ(rf: Tensor, rfmax: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert real RF to tρ ≔ tan(ρ/ρ_max⋅π/2), and θ

    Usage:
        ``lρ, θ = rf2lρθ(rf, rfmax)``
    Inputs:
        - ``rf``: `(N, xy, nT, (nCoils))`, RF pulse, Gauss, x: real, y: imag.
        - ``rfmax``: `(N, (nCoils))`, RF pulse, Gauss.
    Outputs:
        - ``lρ``: `(N, 1, nT, (nCoils))`, logit(ρ/rfmax), [0, +∞).
        - ``θ``: `(N, 1, nT, (nCoils))`, RF phase, [-π, π].

    See Also:
        :func:`~mrphy.utils.lρθ2rf`
    """
    rfmax = rfmax[None] if rfmax.ndim == 0 else rfmax  # scalar to 1d-tensor
    lρ = (rf.norm(dim=1, keepdim=True)/rfmax[:, None, None, ...]).logit()
    θ = torch.atan2(rf[:, [1], :], rf[:, [0], :])
    return lρ, θ


def rf2tρθ(rf: Tensor, rfmax: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert real RF to tρ ≔ tan(ρ/ρ_max⋅π/2), and θ

    Usage:
        ``tρ, θ = rf2tρθ(rf, rfmax)``
    Inputs:
        - ``rf``: `(N, xy, nT, (nCoils))`, RF pulse, Gauss, x: real, y: imag.
        - ``rfmax``: `(N, (nCoils))`, RF pulse, Gauss, x for real, y for imag.
    Outputs:
        - ``tρ``: `(N, 1, nT, (nCoils))`, tan(ρ/rfmax*π/2), [0, +∞).
        - ``θ``: `(N, 1, nT, (nCoils))`, RF phase, [-π, π).

    See Also:
        :func:`~mrphy.utils.tρθ2rf`
    """
    rfmax = rfmax[None] if rfmax.ndim == 0 else rfmax  # scalar to 1d-tensor
    tρ = (rf.norm(dim=1, keepdim=True)/rfmax[:, None, None, ...]*π/2).tan()
    θ = torch.atan2(rf[:, [1], :], rf[:, [0], :])
    return tρ, θ


def rfclamp(rf: Tensor, rfmax: Tensor, *, eps: Number = 1e-7) -> Tensor:
    r"""Clamp RF to rfmax

    Usage:
        ``rf = rfclamp(rf, rfmax, *, eps)``
    Inputs:
        - ``rf``: `(N, xy, nT, (nCoils))`, RF pulse, Gauss, x for real, y for \
          imag.
        - ``rfmax``: `(N, (nCoils))`, RF pulse, Gauss, x for real, y for imag.
    Optionals:
        - ``eps``: effective `rfmax`, is `rfmax-eps`, numerical precession.
    Outputs:
        - ``rf``: `(N, xy, nT, (nCoils))`, \|RF\| clampled at rfmax

    See Also:
        :func:`~mrphy.utils.sclamp`
    """
    rfmax = rfmax[None] if rfmax.ndim == 0 else rfmax
    rf_abs = rf.norm(dim=1, keepdim=True)
    return rf.mul(((rfmax[:, None, None, ...]-eps)/rf_abs).clamp_(max=1))


def s2g(s: Tensor, dt: Tensor = dt0) -> Tensor:
    r"""Compute gradients from slew rates.

    Usage:
        ``g = s2g(s, dt)``

    Inputs:
        - ``s``: `(N, xyz, nT)`, "Gauss/cm/Sec", Slew rate.
    Optionals:
        - ``dt``: `()` ⊻ `(N ⊻ 1,)`, "Sec", dwell time.
    Outputs:
        - ``g``: `(N, xyz, nT)`, "Gauss/cm", Gradient.

    See Also:
        :func:`~mrphy.utils.g2s`
    """
    dt = dt.reshape(dt.shape+(s.ndim-dt.ndim)*(1,))

    g = dt*torch.cumsum(s, dim=2)
    return g


def s2ts(s: Tensor, smax: Tensor) -> Tensor:
    r"""Convert slew rate to ts ≔ tan(s/s_max⋅π/2)

    Usage:
        ``ts = s2ts(s, smax)``
    Inputs:
        - ``s``: `(N, xyz, nT)`, slew rate, Gauss/cm/Sec.
        - ``smax``: `(N, xyz)`, max \|slew rate\|, Gauss/cm/Sec.
    Outputs:
        - ``ts``: `(N, xyz, nT)`, tan(s/smax*π/2), (-∞, ∞)

    See Also:
        :func:`~mrphy.utils.ts2s`
    """
    return (s/smax[..., None]*π/2).tan()


def sclamp(s: Tensor, smax: Tensor) -> Tensor:
    r"""Clamp slew rate to `smax`

    Usage:
        ``s = sclamp(s, smax)``
    Inputs:
        - ``s``: `(N, xyz, nT)`, slew rate, Gauss/cm/Sec.
        - ``smax``: `(N, xyz)`, max \|slew rate\|, Gauss/cm/Sec.
    Outputs:
        - ``s``: `(N, xyz, nT)`, slew rate clamped at smax

    See Also:
        :func:`~mrphy.utils.rfclamp`
    """
    smax = (smax[None] if smax.ndim == 0 else smax).to(s)  # device & dtype
    return s.max(-smax[..., None]).min(smax[..., None])


def ts2s(ts: Tensor, smax: Tensor) -> Tensor:
    r"""Convert ts ≔ tan(s/s_max⋅π/2) to slew rate

    Usage:
        ``s = ts2s(ts, smax)``
    Inputs:
        - ``ts``: `(N, xyz, nT)`, tan(s/smax*π/2), (-∞, ∞)
        - ``smax``: `(N, xyz)`, max \|slew rate\|, Gauss/cm/Sec.
    Outputs:
        - ``s``: `(N, xyz, nT)`, slew rate, Gauss/cm/Sec.

    See Also:
        :func:`~mrphy.utils.s2ts`
    """
    return ts.atan()/π*2*smax[..., None]


def tρθ2rf(tρ: Tensor, θ: Tensor, rfmax: Tensor) -> Tensor:
    r"""Convert tρ ≔ tan(ρ/ρ_max⋅π/2), and θ to real RF

    Usage:
        ``rf = tρθ2rf(tρ, θ, rfmax)``
    Inputs:
        - ``tρ``: `(N, 1, nT, (nCoils))`, tan(ρ/rfmax*π/2), [0, +∞).
        - ``θ``: `(N, 1, nT, (nCoils))`, RF phase, [-π/2, π/2].
        - ``rfmax``: `(N, (nCoils))`, RF pulse, Gauss, x for real, y for imag.
    Outputs:
        - ``rf``: `(N, xy, nT, (nCoils))`, RF pulse, Gauss, x: real, y: imag.

    See Also:
        :func:`~mrphy.utils.rf2tρθ`
    """
    rfmax = rfmax[None] if rfmax.ndim == 0 else rfmax
    rfmax = rfmax[:, None, None, ...]  # -> (N, 1, 1, (nCoils))
    return tρ.atan()/π*2*rfmax*torch.cat((θ.cos(), θ.sin()), dim=1)


def uϕrot(U: Tensor, Φ: Tensor, Vi: Tensor) -> Tensor:
    r"""Rotate Vi about axis U by Φ

    Usage:
        ``Vo = uϕrot(U, Φ, Vi)``

    Apply axis-angle, `U-Phi` rotation on `V`.
    Rotation is broadcasted on `V`.
    See `wikipedia <https://w.wiki/Knf>`_.

    Inputs:
        - ``U``:  `(N, *Nd, xyz)`, 3D rotation axes, assumed unitary;
        - ``Φ``:  `(N, *Nd,)`, rotation angles;
        - ``Vi``: `(N, *Nd, xyz, (nV))`, vectors to be rotated;
    Outputs:
        - ``Vo``: `(N, *Nd, xyz, (nV))`, vectors rotated;
    """
    # No in-place op, repetitive alloc is nece. for tracking the full Jacobian.
    (dim, Φ, U) = ((-1, Φ[..., None], U) if Vi.ndim == U.ndim else
                   (-2, Φ[..., None, None], U[..., None]))

    cΦ, sΦ = torch.cos(Φ), torch.sin(Φ)

    Vo = (cΦ*Vi + (1-cΦ)*torch.sum(U*Vi, dim=dim, keepdim=True)*U
          + sΦ*torch.cross(U.expand_as(Vi), Vi, dim=dim))

    return Vo
