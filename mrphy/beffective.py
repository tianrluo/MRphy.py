r"""B-effective related functions"""

import torch
import torch.nn.functional as F
from torch import tensor, Tensor
from typing import Optional, Tuple

from mrphy import γH, dt0, π
from mrphy import utils

# TODO:
# - Faster init of AB in `beff2ab`


__all__ = ['beff2ab', 'beff2uφ', 'rfgr2beff']


def beff2uϕ(beff: Tensor, γ2πdt: Tensor, *, dim=-1) -> Tuple[Tensor, Tensor]:
    r"""Compute rotation axes and angles from B-effectives

    Usage:
        ``U, Φ = beff2uϕ(beff, γ2πdt, *, dim)``
    Inputs:
        - ``beff``: `(N, *Nd, xyz)`, "Gauss", B-effective, magnetic field \
          applied on `M`.
        - ``γ2πdt``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "rad/Gauss", gyro ratio \
          in radiance mutiplied by `dt`.
    Optionals:
        - ``dim``: int. Indicate the `xyz`-dim, allow \
          `beff.shape != (N, *Nd, xyz)`
    Outputs:
        - ``U``: `(N, *Nd, xyz)`, rotation axis
        - ``Φ``: `(N, *Nd)`, rotation angle
    """
    U = F.normalize(beff, dim=dim)
    Φ = -torch.norm(beff, dim=dim) * γ2πdt  # negate: BxM -> MxB
    return U, Φ


def beff2ab(
    beff: Tensor, *,
    E1: Tensor = tensor(0.),
    E2: Tensor = tensor(0.),
    γ: Tensor = γH,
    dt: Tensor = dt0,
) -> Tuple[Tensor, Tensor]:
    r"""Compute Hargreave's 𝐴/𝐵, mat/vec, from B-effectives

    See: `doi:10.1002/mrm.1170 <https://doi.org/10.1002/mrm.1170>`_.

    Usage:
        ``A, B = beff2ab(beff, *, E1, E2, γ, dt)``

    Inputs:
        - ``beff``: `(N,*Nd,nT,xyz)`, B-effective.
    Optionals:
        - ``T1``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T1 relaxation.
        - ``T2``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T2 relaxation.
        - ``γ``:  `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Hz/Gauss", gyro ratio.
        - ``dt``: `()` ⊻ `(N ⊻ 1,)`, "Sec", dwell time.
    Outputs:
        - ``A``: `(N, *Nd, xyz, 3)`, `A[:,iM,:,:]`, is the `iM`-th 𝐴.
        - ``B``: `(N, *Nd, xyz)`, `B[:,iM,:]`, is the `iM`-th 𝐵.
    """
    shape = beff.shape
    device, dtype, ndim = beff.device, beff.dtype, beff.ndim-2

    dkw = {'device': device, 'dtype': dtype}
    E1, E2, γ, dt = (x.to(device) for x in (E1, E2, γ, dt))

    # reshaping
    E1, E2, γ, dt = (x.reshape(x.shape+(ndim-x.ndim)*(1,))
                     for x in (E1, E2, γ, dt))  # (N, *Nd) compatible

    E1, E2, γ2πdt = E1[..., None], E2[..., None, None], 2*π*γ*dt
    E1_1 = E1.squeeze(dim=-1) - 1

    # C/Python `reshape/view` is different from Fortran/MatLab/Julia `reshape`
    NNd, nT = shape[0:-2], shape[-2]
    s1, s0 = NNd+(1, 1), NNd+(1, 4)

    AB = torch.cat([torch.ones(s1, **dkw), torch.zeros(s0, **dkw),
                    torch.ones(s1, **dkw), torch.zeros(s0, **dkw),
                    torch.ones(s1, **dkw), torch.zeros(s1, **dkw)],
                   dim=-1).view(NNd+(3, 4))  # -> (N, *Nd, xyz, 3+1)

    # simulation
    for t in range(nT):
        u, ϕ = beff2uϕ(beff[..., t, :], γ2πdt)

        if torch.any(ϕ != 0):
            AB1 = utils.uϕrot(u, ϕ, AB)
        else:
            AB1 = AB

        # Relaxation
        AB1[..., 0:2, :] *= E2
        AB1[..., 2, :] *= E1
        AB1[..., 2, 3] -= E1_1
        AB, AB1 = AB1, AB

    A, B = AB[..., 0:3], AB[..., 3]

    return A, B


def rfgr2beff(
    rf: Tensor,
    gr: Tensor,
    loc: Tensor, *,
    Δf: Optional[Tensor] = None,
    b1Map: Optional[Tensor] = None,
    γ: Tensor = γH
) -> Tensor:
    r"""Compute B-effectives from rf and gradients

    Usage:
        ``beff = rfgr2beff(rf, gr, loc, *, Δf, b1Map, γ)``
    Inputs:
        - ``rf``: `(N,xy,nT,(nCoils))`, "Gauss", `xy` for separating real and \
          imag part.
        - ``gr``: `(N,xyz,nT)`, "Gauss/cm".
        - ``loc``: `(N,*Nd,xyz)`, "cm", locations.
    Optionals:
        - ``Δf``: `(N,*Nd,)`, "Hz", off-resonance.
        - ``b1Map``: `(N, *Nd, xy (, nCoils)`, a.u., transmit sensitivity.
        - ``γ``:  `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Hz/Gauss", gyro ratio.
    Outputs:
        - ``beff``: `(N,*Nd,nT,xyz)`, "Gauss"
    """
    assert(rf.device == gr.device == loc.device)
    device = rf.device

    shape = loc.shape
    N, Nd, ndim = shape[0], shape[1:-1], loc.ndim-2

    Bz = (loc.reshape(N, -1, 3) @ gr).reshape((N, *Nd, -1))

    if Δf is not None:  # Δf: -> (N, *Nd, 1); 3 from 1(dim-N) + 2(dim-xtra)
        γ = γ.to(device=device)
        Δf, γ = (_.reshape(_.shape+(ndim+2-_.ndim)*(1,)) for _ in (Δf, γ))
        Bz += Δf/γ

    # rf -> (N, *len(Nd)*(1,), xy, nT, (nCoils))
    rf = rf.reshape((-1,) + ndim*(1,) + rf.shape[1:])
    # Real as `Bx`, Imag as `By`.
    if b1Map is None:
        if rf.ndim == Bz.ndim+2:  # (N, *len(Nd)*(1,), xy, nT, nCoils)
            rf = torch.sum(rf, dim=-1)  # -> (N, *len(Nd)*(1,), xy, nT)

        Bx, By = rf[..., 0, :].expand_as(Bz), rf[..., 1, :].expand_as(Bz)
    else:
        if b1Map.ndim == 1+len(Nd)+1:
            b1Map = b1Map[..., None]  # (N, *Nd, xy) -> (N, *Nd, xy, 1)
        if rf.ndim == b1Map.ndim:  # rf missing the nCoil dim
            rf = rf[..., None]

        b1Map = b1Map.to(device)
        b1Map = b1Map[..., None, :]  # -> (N, *Nd, xy, 1, nCoils)
        Bx = torch.sum((b1Map[..., 0, :, :]*rf[..., 0, :, :]
                        - b1Map[..., 1, :, :]*rf[..., 1, :, :]),
                       dim=-1).expand_as(Bz)  # -> (N, *Nd, x, nT)
        By = torch.sum((b1Map[..., 0, :, :]*rf[:, :, 1, ...]
                        + b1Map[..., 1, :, :]*rf[:, :, 0, ...]),
                       dim=-1).expand_as(Bz)  # -> (N, *Nd, y, nT)

    beff = torch.stack([Bx, By, Bz], dim=-1)  # -> (N, *Nd, nT, xyz)
    return beff
