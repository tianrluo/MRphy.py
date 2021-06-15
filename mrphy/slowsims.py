r"""Simulation codes with implicit Jacobian operations.
"""

import torch
from torch import tensor, Tensor
from typing import Optional, Tuple

from mrphy import γH, dt0, π
from mrphy import utils, beffective


__all__ = ['blochsim_1step', 'blochsim', 'blochsim_ab', 'freeprec']


def blochsim_1step(
    M: Tensor, M1: Tensor, b: Tensor,
    E1: Tensor, E1_1: Tensor, E2: Tensor, γ2πdt: Tensor
) -> Tuple[Tensor, Tensor]:
    r"""Single step bloch simulation

    Usage:
        ``M = blochsim_1step(M, M1, b, E1, E1_1, E2, γ2πdt)``
    Inputs:
        - ``M``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
          [[[0 0 1]]].
        - ``M1``: `(N, *Nd, xyz)`, pre-allocated variable for `uϕrot` output.
        - ``b``: `(N, *Nd, xyz)`, "Gauss", B-effective, magnetic field applied.
        - ``E1``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, a.u., T1 reciprocal exponential.
        - ``E1_1``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, a.u., T1 reciprocal \
          exponential subtracted by ``1``.
        - ``E2``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, a.u., T2 reciprocal exponential.
        - ``γ2πdt``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "rad/Gauss", gyro ratio \
          in radiance mutiplied by `dt`.
    Outputs:
        - ``M``: `(N, *Nd, xyz)`, Magetic spins after simulation.
    """
    u, ϕ = beffective.beff2uϕ(b, γ2πdt)

    if torch.any(ϕ != 0):
        M1 = utils.uϕrot(u, ϕ, M)
    else:
        M1 = M
    # Relaxation
    M1[..., 0:2] *= E2[..., None]
    M1[..., 2] *= E1
    M1[..., 2] -= E1_1

    M, M1 = M1, M
    return M, M1


def blochsim(
    M: Tensor, Beff: Tensor, *,
    T1: Optional[Tensor] = None, T2: Optional[Tensor] = None,
    γ: Tensor = γH, dt: Tensor = dt0
) -> Tensor:
    r"""Bloch simulator with implicit Jacobian operations.

    Usage:
        ``Mo = blochsim(Mi, Beff, *, T1, T2, γ, dt)``
        ``Mo = blochsim(Mi, Beff, *, T1=None, T2=None, γ, dt)``
    Inputs:
        - ``M``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
          [[[0 0 1]]].
        - ``Beff``: `(N, *Nd, xyz, nT)`, "Gauss", B-effective, magnetic field.
    OPTIONALS:
        - ``T1``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T1 relaxation.
        - ``T2``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T2 relaxation.
        - ``γ``:  `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Hz/Gauss", gyro ratio.
        - ``dt``: `()` ⊻ `(N ⊻ 1,)`, "Sec", dwell time.
    Outputs:
        - ``M``: `(N, *Nd, xyz)`, Magetic spins after simulation.

    .. note::
        spin history during simulations is not provided.
    """
    assert(M.shape[:-1] == Beff.shape[:-2])
    device, dtype, ndim = M.device, M.dtype, M.ndim-1

    # defaults and move to the same device
    dkw = {'device': device, 'dtype': dtype}
    E1 = tensor(1, **dkw) if (T1 is None) else torch.exp(-dt/T1.to(device))
    E2 = tensor(1, **dkw) if (T2 is None) else torch.exp(-dt/T2.to(device))
    Beff, γ, dt = (x.to(device) for x in (Beff, γ, dt))

    # preprocessing
    E1, E2, γ, dt = map(lambda x: x.reshape(x.shape+(ndim-x.ndim)*(1,)),
                        (E1, E2, γ, dt))  # (N, *Nd) compatible

    E1_1, E2, γ2πdt = E1 - 1, E2[..., None], 2*π*γ*dt  # Hz/Gs -> Rad/Gs

    # simulation
    for t in range(Beff.shape[-1]):
        u, ϕ = beffective.beff2uϕ(Beff[..., t], γ2πdt)
        if torch.any(ϕ != 0):
            M1 = utils.uϕrot(u, ϕ, M)
        else:
            M1 = M
        # Relaxation
        M1[..., 0:2] *= E2
        M1[..., 2] *= E1
        M1[..., 2] -= E1_1

        M, M1 = M1, M

    return M


def blochsim_ab(M: Tensor, A: Tensor, B: Tensor) -> Tensor:
    r"""Bloch simulation via Hargreave's mat/vec representation

    Usage:
        ``M = blochsim_ab(M, A, B)``
    Inputs:
        - ``M``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
          magnitude [0 0 1]
        - ``A``: `(N, *Nd, xyz, 3)`, ``A[:,iM,:,:]`` is the `iM`-th 𝐴.
        - ``B``: `(N, *Nd, xyz)`, ``B[:,iM,:]`` is the `iM`-th 𝐵.
    Outputs:
        - ``M``: `(N, *Nd, xyz)`, Result magnetic spins
    """
    M = (A @ M[..., None]).squeeze_(dim=-1) + B
    return M


def freeprec(
    M: Tensor, dur: Tensor, *,
    T1: Optional[Tensor] = None, T2: Optional[Tensor] = None,
    Δf: Optional[Tensor] = None
) -> Tensor:
    r"""Isochromats free precession with given relaxation and off-resonance

    Usage:
        ``M = freeprec(M, dur, *, T1, T2, Δf)``
    Inputs:
        - ``M``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
          magnitude [0 0 1]
        - ``dur``: `()` ⊻ `(N ⊻ 1,)`, "Sec", duration of free-precession.
    OPTIONALS:
        - ``T1``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T1 relaxation.
        - ``T2``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T2 relaxation.
        - ``Δf``: `(N ⊻ 1, *Nd ⊻ 1,)`, "Hz", off-resonance.
    Outputs:
        - ``M``: `(N, *Nd, xyz)`, Result magnetic spins
    """
    ndim = M.ndim  # dur, T1, T2, Δf are reshaped to be compatible w/ M
    dur = dur.reshape(dur.shape+(ndim-dur.ndim)*(1,))

    Mx, My, Mz = M.split(1, dim=-1)  # (N, *Nd, 1)

    # Precession
    if Δf is not None:
        Δf = Δf.reshape(Δf.shape+(ndim-Δf.ndim)*(1,))
        ϕ = -(2*π)*Δf*dur  # positive Δf dephases spin clock-wise/negatively
        cϕ, sϕ = torch.cos(ϕ), torch.sin(ϕ)
        Mx, My = cϕ*Mx-sϕ*My, sϕ*Mx+cϕ*My

    # Relaxation
    assert((T1 is None) == (T2 is None))  # both or neither
    if T1 is not None:
        T1, T2 = (x.reshape(x.shape+(ndim-x.ndim)*(1,)) for x in (T1, T2))
        E1, E2 = torch.exp(-dur/T1), torch.exp(-dur/T2)
        Mx, My, Mz = E2*Mx, E2*My, E1*Mz+1-E1

    M = torch.cat((Mx, My, Mz), dim=-1)  # (N, *Nd, xyz)
    return M
