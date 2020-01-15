import numpy as np
from numpy import ndarray
import torch
from torch import tensor, Tensor

from mrphy import γH, dt0


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
    """
    assert((not isTx) or torch.all(k[:, :, -1] == 0))  # Tx k must end at 0
    g = torch.cat((k[:, :, [0]],
                   k[:, :, 1:] - k[:, :, :-1]),
                  dim=2)/((γ*dt)[..., None])
    return g


def rf_c2r(rf: ndarray) -> ndarray:
    """
        rf_c2r(rf)
    *INPUTS*:
    - `rf` (N, 1, nT, (nCoils)) RF pulse, complex
    *OUTPUTS*:
    - `rf` (N, xy, nT, (nCoils)) RF pulse, x for real, y for imag.
    """
    return np.concatenate((np.real(rf), np.imag(rf)), axis=1)


def rf_r2c(rf: ndarray) -> ndarray:
    """
        rf_r2c(rf)
    *INPUTS*:
    - `rf` (N, xy, nT, (nCoils)) RF pulse, x for real, y for imag.
    *OUTPUTS*:
    - `rf` (N, 1, nT, (nCoils)) RF pulse, complex.
    """
    return rf[:, 0, ...] + 1j*rf[:, 1, ...]


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
