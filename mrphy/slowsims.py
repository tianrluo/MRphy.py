import torch
from torch import tensor, Tensor
from typing import Optional

from mrphy import γH, dt0, π
from mrphy import utils, beffective

# TODO:
# - Allow Vo to be allocated outside `beff2uϕ`, `uϕrot` and `rfgr2beff`


def blochsim_1step(
        M: Tensor, M1: Tensor, b: Tensor,
        E1: Tensor, E1_1: Tensor, E2: Tensor, γ2πdt: Tensor):
    """
        blochsim_1step(M, M1, b, E1, E1_1, E2, γ2πdt)
    *INPUTS*:
    - `M` (N, *Nd, xyz), Magnetic spins, assumed equilibrium magnitude [0 0 1]
    - `M1` (N, *Nd, xyz), pre-allocated variable for `uϕrot` output.
    - `b` (N, *Nd, xyz) "Gauss", B-effective, magnetic field applied.
    - `E1` (N, 1,) a.u., T1 reciprocal exponential, global.
    - `E1_1` (N, 1,) a.u., T1 reciprocal exponential subtracted by `1`, global.
    - `E2` (N, 1,) a.u., T2 reciprocal exponential, global.
    - `γ2πdt` (N, 1,) "rad/Gauss", gyro ratio mutiplied by `dt`, global.
    *OUTPUTS*:
    - `M` (N, *Nd, xyz), Magetic spins after simulation.
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
        M: Tensor, Beff: Tensor,
        T1: Optional[Tensor] = None, T2: Optional[Tensor] = None,
        γ: Optional[Tensor] = None, dt: Optional[Tensor] = None):
    """
    *INPUTS*:
    - `M` (N, *Nd, xyz), Magnetic spins, assumed equilibrium magnitude [0 0 1]
    - `Beff` (N, *Nd, xyz, nT) "Gauss", B-effective, magnetic field applied.
    *OPTIONALS*:
    - `T1` (N, *Nd,) "Sec", T1 relaxation.
    - `T2` (N, *Nd,) "Sec", T2 relaxation.
    - `γ`  (N, *Nd,) "Hz/Gauss", gyro ratio in Hertz.
    - `dt` (N, 1, ) "Sec", dwell time.
    *OUTPUTS*:
    - `M` (N, *Nd, xyz), Magetic spins after simulation.
    *Notes*:
      spin history during simulations is not provided at the moment.
    """
    assert(M.shape[:-1] == Beff.shape[:-2])
    device, dtype, d = M.device, M.dtype, M.dim()-1

    # defaults and move to the same device
    Beff = Beff.to(device)
    dkw = {'device': device, 'dtype': dtype}
    dt = tensor(dt0, **dkw) if (dt0 is None) else dt.to(device)
    γ = tensor(γH, **dkw) if (γ is None) else γ.to(device)
    E1 = tensor(1, **dkw) if (T1 is None) else torch.exp(-dt/T1.to(device))
    E2 = tensor(1, **dkw) if (T2 is None) else torch.exp(-dt/T2.to(device))

    # preprocessing
    E1, E2, γ, dt = map(lambda x: x.reshape(x.shape+(d-x.dim())*(1,)),
                        (E1, E2, γ, dt))  # broadcastable w/ (N, *Nd)

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


def blochsim_ab(M: Tensor, A: Tensor, B: Tensor):
    """
    *INPUTS*:
    - `M` (N, *Nd, xyz), Magnetic spins, assumed equilibrium magnitude [0 0 1]
    - `A` (N, *Nd, xyz, 3), `A[:,iM,:,:]` is the `iM`-th 𝐴.
    - `B` (N, *Nd, xyz), `B[:,iM,:]` is the `iM`-th 𝐵.
    *INPUTS*:
    - `M` (N, *Nd, xyz), Result magnetic spins
    """
    M = (A @ M[..., None]).squeeze_(dim=-1) + B
    return M
