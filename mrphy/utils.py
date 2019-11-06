import torch
from torch import tensor

from mrphy import γH, dt0


def ctrsub(shape):
    """
        ctrsub(shape)
    Center index after fftshift, wrapped for consistent behaviours.
    *OUTPUTS*
    - `cSub`
    """
    return shape//2


def k2g(k: torch.Tensor,
        isTx: bool,
        γ: torch.Tensor = tensor([[γH]]),
        dt: torch.Tensor = tensor([[dt0]])):
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


def g2k(g: torch.Tensor,
        isTx: bool,
        γ: torch.Tensor = tensor([[γH]]),
        dt: torch.Tensor = tensor([[dt0]])):
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


def g2s(g: torch.Tensor,
        dt: torch.Tensor = tensor([[dt0]])):
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
