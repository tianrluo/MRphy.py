import torch
import torch.nn.functional as F
from torch import tensor, Tensor

from mrphy import γH, dt0, π
from mrphy import utils

# TODO:
# - Faster init of AB in `beff2ab`


def rfgr2beff(
        rf: Tensor, gr: Tensor, loc: Tensor,
        Δf: Tensor = None, b1Map: Tensor = None, γ: Tensor = γH):
    """
        beff = rfgr2beff(rf, gr, loc, Δf, b1Map, γ)
    *INPUTS*:
    - `rf` (N,xy, nT,(nCoils)) "Gauss", `xy` for separating real and imag part.
    - `gr` (N,xyz,nT) "Gauss/cm"
    *OPTIONALS*:
    - `loc`(N,*Nd,xyz) "cm", locations.
    - `Δf` (N,*Nd,) "Hz", off-resonance.
    - `b1Map` (N,*Nd,xy,nCoils) a.u., , transmit sensitivity.
    - `γ`(N,1) "Hz/Gauss", gyro-ratio
    *OUTPUTS*:
    - `beff` (N,*Nd,xyz,nT) "Gauss"
    """
    assert(rf.device == gr.device == loc.device)
    device = rf.device

    shape = loc.shape
    N, Nd, d = shape[0], shape[1:-1], loc.dim()-2

    Bz = (loc.reshape(N, -1, 3) @ gr).reshape((N, *Nd, 1, -1))

    if Δf is not None:  # Δf: -> (N, *Nd, 1, 1); 3 from 1(dim-N) + 2(dim-xtra)
        γ = γ.to(device=device)
        Δf, γ = map(lambda x: x.reshape(x.shape+(d+3-x.dim())*(1,)), (Δf, γ))
        Bz += Δf/γ

    # rf -> (N, *len(Nd)*(1,), xy, nT, (nCoils))
    rf = rf.reshape((-1, *d*(1,))+rf.shape[1:])
    # Real as `Bx`, Imag as `By`.
    if b1Map is None:
        if rf.dim() == Bz.dim()+1:  # (N, *len(Nd)*(1,), xy, nT, nCoils)
            rf = torch.sum(rf, dim=-1)  # -> (N, *len(Nd)*(1,), xy, nT)

        Bx, By = rf[..., 0:1, :].expand_as(Bz), rf[..., 1:2, :].expand_as(Bz)
    else:
        b1Map = b1Map.to(device)
        b1Map = b1Map[..., None, :]  # -> (N, *Nd, xy, 1, nCoils)
        Bx = torch.sum((b1Map[..., 0:1, :, :]*rf[..., 0:1, :, :]
                        - b1Map[..., 1:2, :, :]*rf[..., 1:2, :, :]),
                       dim=-1).expand_as(Bz)  # -> (N, *Nd, x, nT)
        By = torch.sum((b1Map[..., 0:1, :, :]*rf[:, :, 1:2, ...]
                        + b1Map[..., 1:2, :, :]*rf[:, :, 0:1, ...]),
                       dim=-1).expand_as(Bz)  # -> (N, *Nd, y, nT)

    beff = torch.cat([Bx, By, Bz], dim=-2)  # -> (N, *Nd, xyz, nT)
    return beff


def beff2uϕ(beff: Tensor, γ2πdt: Tensor, dim=-1):
    """
        U, Φ = beff2uϕ(beff, γ2πdt)
    *INPUTS*:
    - `beff` (N, *Nd, xyz) "Gauss", B-effective, magnetic field applied on `M`.
    - `γ2πdt` (N, 1,) "Rad/Gauss", gyro ratio in radians, global.
    *OPTIONALS*
    - `dim` int. Indicate the `xyz`-dim, allow `beff.shape != (N, *Nd, xyz)`
    *OUTPUTS*:
    - `U` (N, *Nd, xyz), rotation axis
    - `Φ` (N, *Nd), rotation angle
    """
    U = F.normalize(beff, dim=dim)
    Φ = -torch.norm(beff, dim=dim) * γ2πdt  # negate: BxM -> MxB
    return U, Φ


def beff2ab(
        beff: Tensor,
        E1: Tensor = None, E2: Tensor = None,
        γ: Tensor = None, dt: Tensor = None):
    """
        beff2ab(beff, T1=(Inf), T2=(Inf), γ=γ¹H, dt=(dt0))
    Turn B-effective into Hargreave's 𝐴/𝐵, mat/vec, see: doi:10.1002/mrm.1170.

    *INPUTS*:
    - `beff`: (N,*Nd,xyz,nT).
    *OPTIONALS*:
    - `T1` (N, *Nd,) "Sec", T1 relaxation.
    - `T2` (N, *Nd,) "Sec", T2 relaxation.
    - `γ`  (N, *Nd,) "Hz/Gauss", gyro ratio in Hertz.
    - `dt` (N, 1, ) "Sec", dwell time.
    *OUTPUTS*:
    - `A` (N, *Nd, xyz, 3), `A[:,iM,:,:]` is the `iM`-th 𝐴.
    - `B` (N, *Nd, xyz), `B[:,iM,:]` is the `iM`-th 𝐵.
    """
    shape = beff.shape
    device, dtype, d = beff.device, beff.dtype, beff.dim()-2

    # defaults
    dkw = {'device': device, 'dtype': dtype}
    dt = tensor(dt0, **dkw) if (dt0 is None) else dt.to(device)
    γ = tensor(γH, **dkw) if (γ is None) else γ.to(device)
    E1 = tensor(0, **dkw) if (E1 is None) else E1.to(device)
    E2 = tensor(0, **dkw) if (E2 is None) else E2.to(device)

    # reshaping
    E1, E2, γ, dt = map(lambda x: x.reshape(x.shape+(d-x.dim())*(1,)),
                        (E1, E2, γ, dt))  # broadcastable w/ (N, *Nd)

    E1, E2, γ2πdt = E1[..., None], E2[..., None, None], 2*π*γ*dt
    E1_1 = E1.squeeze(dim=-1) - 1

    # C/Python `reshape/view` is different from Fortran/MatLab/Julia `reshape`
    NNd, nT = shape[0:-2], shape[-1]
    s1, s0 = NNd+(1, 1), NNd+(1, 4)

    AB = torch.cat([torch.ones(s1, **dkw), torch.zeros(s0, **dkw),
                    torch.ones(s1, **dkw), torch.zeros(s0, **dkw),
                    torch.ones(s1, **dkw), torch.zeros(s1, **dkw)],
                   dim=-1).view(NNd+(3, 4))  # -> (N, *Nd, xyz, 3+1)

    # simulation
    for t in range(nT):
        u, ϕ = beff2uϕ(beff[..., t], γ2πdt)

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
