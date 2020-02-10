from typing import Tuple, Union
from numbers import Number

import torch
import numpy as np
from numpy import ndarray as ndarray_c
from torch import tensor, Tensor

from mrphy import γH, dt0, π
if torch.cuda.is_available():
    import cupy as cp
    from cupy import ndarray as ndarray_g
    ndarrayA = Union[ndarray_c, ndarray_g]
else:
    ndarrayA = ndarray_c


def ctrsub(shape):
    """
        ctrsub(shape)
    Center index after fftshift, wrapped for consistent behaviours.
    *OUTPUTS*
    - `cSub`
    """
    return shape//2


def g2k(g: Tensor, isTx: bool,
        γ: Tensor = tensor([[γH]]), dt: Tensor = tensor([[dt0]])) -> Tensor:
    """
        g2k(g, isTx, γ=γ¹H, dt=dt0)
    Compute k-space from gradient.

    *INPUTS*:
    - `g` (N, xyz, nT) "Gauss/cm", gradient
    - `isTx`, if `true`, compute transmit k-space, `k`, ends at the origin.
    *OPTIONALS*:
    - `γ` (N, 1,) "Hz/Gauss", gyro-ratio.
    - `dt` (N, 1,) "sec", gradient temporal step size, i.e., dwell time.
    *OUTPUTS*:
    - `k` (N, xyz, nT) "cycle/cm", Tx or Rx k-space.

    See Also:
    `g2s`, `k2g`
    """
    k = γ * dt * torch.cumsum(g, dim=2)
    if isTx:
        k -= k[:, :, [-1]]
    return k


def g2s(g: Tensor, dt: Tensor = tensor([[dt0]])) -> Tensor:
    """
    *INPUTS*:
    - `g` (N, xyz, nT) "Gauss/cm", gradient
    *OPTIONALS*:
    - `dt` (N, 1,) "sec", gradient temporal step size, i.e., dwell time.
    *OUTPUTS*:
    - `s` (N, xyz, nT) "cycle/cm/sec", slew rate

    See Also:
    `g2k`, `s2g`
    """
    s = torch.cat((g[:, :, [0]],
                   g[:, :, 1:] - g[:, :, :-1]),
                  dim=2)/(dt[..., None])
    return s


def k2g(k: Tensor, isTx: bool,
        γ: Tensor = tensor([[γH]]), dt: Tensor = tensor([[dt0]])) -> Tensor:
    """
        k2g(k, isTx, γ=γ¹H, dt=dt0)
    Gradient, `g`, of `TxRx` k-space, (trasmit/receive, excitation/imaging).

    *INPUTS*:
    - `k` (N, xyz, nT) "cycle/cm", Tx or Rx k-space.
    - `isTx`, if `true`, compute transmit k-space, `k`, ends at the origin.
    *OPTIONALS*:
    - `γ` (N, 1,) "Hz/Gauss", gyro-ratio.
    - `dt` (N, 1,) "sec", gradient temporal step size, i.e., dwell time.
    *OUTPUTS*:
    - `g` (N, xyz, nT) "Gauss/cm", gradient

    See Also:
    `g2k`
    """
    assert((not isTx) or torch.all(k[:, :, -1] == 0))  # Tx k must end at 0
    g = torch.cat((k[:, :, [0]],
                   k[:, :, 1:] - k[:, :, :-1]),
                  dim=2)/((γ*dt)[..., None])
    return g


def rf_c2r(rf: ndarrayA) -> ndarrayA:
    """
        rf_c2r(rf)
    *INPUTS*:
    - `rf` (N, 1, nT, (nCoils)) RF pulse, complex
    *OUTPUTS*:
    - `rf` (N, xy, nT, (nCoils)) RF pulse, x for real, y for imag.

    See Also:
    `rf_r2c`
    """
    if isinstance(rf, ndarray_c):
        return np.concatenate((np.real(rf), np.imag(rf)), axis=1)
    else:  # ndarray_g, i.e., cupy.ndarray
        return cp.concatenate((cp.real(rf), cp.imag(rf)), axis=1)


def rf_r2c(rf: ndarrayA) -> ndarrayA:
    """
        rf_r2c(rf)
    *INPUTS*:
    - `rf` (N, xy, nT, (nCoils)) RF pulse, x for real, y for imag.
    *OUTPUTS*:
    - `rf` (N, 1, nT, (nCoils)) RF pulse, complex.

    See Also:
    `rf_c2r`
    """
    return rf[:, [0], ...] + 1j*rf[:, [1], ...]


def rf2tρθ(rf: Tensor, rfmax: Tensor) -> Tuple[Tensor, Tensor]:
    """
        rf2tρθ(rf, rfmax)
    *INPUTS*:
    - `rf` (N, xy, nT, (nCoils)) RF pulse, Gauss, x for real, y for imag.
    - `rfmax` (N, (nCoils)) RF pulse, Gauss, x for real, y for imag.
    *OUTPUTS*:
    - `tρ` (N, 1, nT, (nCoils)) tan(ρ/rfmax*π/2), [0, +∞).
    - `θ` (N, 1, nT, (nCoils)) RF phase, [-π/2, π/2].

    See Also:
    `tρθ2rf`
    """
    rfmax = rfmax[None] if rfmax.ndim == 0 else rfmax
    tρ = (rf.norm(dim=1, keepdim=True)/rfmax[:, None, None, ...]*π/2).tan()
    θ = torch.atan2(rf[:, [1], :], rf[:, [0], :])
    return tρ, θ


def rfclamp(rf: Tensor, rfmax: Tensor, eps: Number = 1e-7) -> Tensor:
    """
        rfclamp(rf, rfmax)
    *INPUTS*:
    - `rf` (N, xy, nT, (nCoils)) RF pulse, Gauss, x for real, y for imag.
    - `rfmax` (N, (nCoils)) RF pulse, Gauss, x for real, y for imag.
    *OPTIONALS*:
    - `eps` effective `rfmax` is `rfmax-eps`, numerical precession.
    *OUTPUTS*:
    - `rf` (N, xy, nT, (nCoils)) |RF| clampled at rfmax

    See Also:
    `sclamp`
    """
    rfmax = rfmax[None] if rfmax.ndim == 0 else rfmax
    rf_abs = rf.norm(dim=1, keepdim=True)
    return rf.mul(((rfmax[:, None, None, ...]-eps)/rf_abs).clamp_(max=1))


def s2g(s: Tensor, dt: Tensor = tensor([[dt0]])) -> Tensor:
    """
        s2g(s, dt=dt0)
    Compute gradient from slew rate.

    *INPUTS*:
    - `s` (N, xyz, nT) "Gauss/cm/Sec", Slew rate.
    *OPTIONALS*:
    - `dt` (N, 1,) "sec", gradient temporal step size, i.e., dwell time.
    *OUTPUTS*:
    - `g` (N, xyz, nT) "Gauss/cm", Gradient.

    See Also:
    `g2s`
    """
    g = dt[..., None]*torch.cumsum(s, dim=2)
    return g


def s2ts(s: Tensor, smax: Tensor) -> Tensor:
    """
        s2ts(s, smax)
    *INPUTS*:
    - `s` (N, xyz, nT) slew rate, Gauss/cm/Sec.
    - `smax` (N, xyz) max |slew rate|, Gauss/cm/Sec.
    *OUTPUTS*:
    - `ts` (N, xyz, nT) tan(s/smax*π/2), (-∞, ∞)

    See Also:
    `ts2s`
    """
    return (s/smax[..., None]*π/2).tan()


def sclamp(s: Tensor, smax: Tensor) -> Tensor:
    """
        sclamp(s, smax)
    *INPUTS*:
    - `s` (N, xyz, nT) slew rate, Gauss/cm/Sec.
    - `smax` (N, xyz) max |slew rate|, Gauss/cm/Sec.
    *OUTPUTS*:
    - `s` (N, xyz, nT) slew rate clamped at smax

    See Also:
    `rfclamp`
    """
    smax = (smax[None] if smax.ndim == 0 else smax).to(s)  # device & dtype
    return s.max(-smax[..., None]).min(smax[..., None])


def ts2s(ts: Tensor, smax: Tensor) -> Tensor:
    """
        s2ts(s, smax)
    *INPUTS*:
    - `ts` (N, xyz, nT) tan(s/smax*π/2), (-∞, ∞)
    - `smax` (N, xyz) max |slew rate|, Gauss/cm/Sec.
    *OUTPUTS*:
    - `s` (N, xyz, nT) slew rate, Gauss/cm/Sec.

    See Also:
    `s2ts`
    """
    return ts.atan()/π*2*smax[..., None]


def tρθ2rf(tρ: Tensor, θ: Tensor, rfmax: Tensor) -> Tensor:
    """
        tρθ2rf(tρ, θ, rfmax)
    *INPUTS*:
    - `tρ` (N, 1, nT, (nCoils)) tan(ρ/rfmax*π/2), [0, +∞).
    - `θ` (N, 1, nT, (nCoils)) RF phase, [-π/2, π/2].
    - `rfmax` (N, (nCoils)) RF pulse, Gauss, x for real, y for imag.
    *OUTPUTS*:
    - `rf` (N, xy, nT, (nCoils)) RF pulse, Gauss, x for real, y for imag.

    See Also:
    `rf2tρθ`
    """
    rfmax = rfmax[None] if rfmax.ndim == 0 else rfmax
    rfmax = rfmax[:, None, None, ...]  # -> (N, 1, 1, (nCoils))
    return tρ.atan()/π*2*rfmax*torch.cat((θ.cos(), θ.sin()), dim=1)


def uϕrot(U: Tensor, Φ: Tensor, Vi: Tensor):
    """
        Vo = uϕrot(U, Φ, Vi)
    Apply axis-angle, `U-Phi` rotation on `V`. Rotation is broadcasted on `V`.
    <en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle>

    *INPUTS*:
    - `U`  (N, *Nd, xyz), 3D rotation axes, assumed unitary;
    - `Φ`  (N, *Nd,), rotation angles;
    - `Vi` (N, *Nd, xyz, (nV)), vectors to be rotated;
    *OUTPUTS*:
    - `Vo` (N, *Nd, xyz, (nV)), vectors rotated;
    """
    # No in-place op, repetitive alloc is nece. for tracking the full Jacobian.
    (dim, Φ, U) = ((-1, Φ[..., None], U) if Vi.dim() == U.dim() else
                   (-2, Φ[..., None, None], U[..., None]))

    cΦ, sΦ = torch.cos(Φ), torch.sin(Φ)

    Vo = (cΦ*Vi + (1-cΦ)*torch.sum(U*Vi, dim=dim, keepdim=True)*U
          + sΦ*torch.cross(U.expand_as(Vi), Vi, dim=dim))

    return Vo
