import torch
import torch.nn.functional as F
from torch import tensor

from mrphy import γH, dt0, π, inf

# TODO:
# - Faster init of AB in `beff2ab`
# - Allow Vo to be allocated outside `beff2uϕ`, `uϕrot` and `rfgr2beff`


def rfgr2beff(
        rf: torch.Tensor, gr: torch.Tensor, loc: torch.Tensor,
        Δf: torch.Tensor = None, b1Map: torch.Tensor = None,
        γ: torch.Tensor = None):
    """
        Beff = rfgr2beff(rf, gr, loc, Δf, b1Map, γ)
    *INPUTS*:
    - `rf` (N,xy, nT,(nCoils)) "Gauss", `xy` for separating real and imag part.
    - `gr` (N,xyz,nT) "Gauss/cm"
    *OPTIONALS*:
    - `loc`(N,*Nd,xyz) "cm", locations.
    - `Δf` (N,*Nd,) "Hz", off-resonance.
    - `b1Map` (N,*Nd,xy,nCoils) a.u., , transmit sensitivity.
    - `γ`(N,1) "Hz/Gauss", gyro-ratio
    *OUTPUTS*:
    - `Beff`  (N,*Nd,xyz,nT)
    """
    assert(rf.device == gr.device == loc.device)
    device = rf.device

    shape = loc.shape
    N, Nd, d = shape[0], shape[1:-1], loc.dim()-2

    Bz = (loc.reshape(N, -1, 3) @ gr).reshape((N, *Nd, 1, -1))

    if Δf is not None:  # Δf: -> (N, *Nd, 1, 1); 3 from 1(dim-N) + 2(dim-xtra)
        γ = (torch.Tensor([[γH]], device=device, dtype=Δf.dtype)
             if (γ is None) else γ.to(device))
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

    Beff = torch.cat([Bx, By, Bz], dim=-2)  # -> (N, *Nd, xyz, nT)
    return Beff


def beff2uϕ(Beff: torch.Tensor, γ2πdt: torch.Tensor, dim=-1):
    """
        U, Φ = beff2uϕ(Beff, γ2πdt)
    *INPUTS*:
    - `Beff` (N, *Nd, xyz) "Gauss", B-effective, magnetic field applied on `M`.
    - `γ2πdt` (N, 1,) "Rad/Gauss", gyro ratio in radians, global.
    *OPTIONALS*
    - `dim` int. Indicate the `xyz`-dim, allow `Beff.shape != (N, *Nd, xyz)`
    *OUTPUTS*:
    - `U` (N, *Nd, xyz), rotation axis
    - `Φ` (N, *Nd), rotation angle
    """
    U = F.normalize(Beff, dim=dim)
    Φ = -torch.norm(Beff, dim=dim) * γ2πdt  # negate: BxM -> MxB
    return U, Φ


def uϕrot(U: torch.Tensor, Φ: torch.Tensor, Vi: torch.Tensor):
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


def beff2ab(
        Beff: torch.Tensor,
        T1: torch.Tensor = None, T2: torch.Tensor = None,
        γ: torch.Tensor = None, dt: torch.Tensor = None):
    """
        beff2ab(Beff, T1=(Inf), T2=(Inf), γ=γ¹H, dt=(dt0))
    Turn B-effective into Hargreave's 𝐴/𝐵, mat/vec, see: doi:10.1002/mrm.1170.

    *INPUTS*:
    - `Beff`: (N,*Nd,xyz,nT).
    *OPTIONALS*:
    - `T1` (N, 1,) "Sec", T1 relaxation, global.
    - `T2` (N, 1,) "Sec", T2 relaxation, global.
    - `γ` (N, 1,) "Hz/Gauss", gyro ratio in Hertz, global.
    - `dt` (N, 1,) "Sec", dwell time, global.
    *OUTPUTS*:
    - `A` (N, *Nd, xyz, 3), `A[:,iM,:,:]` is the `iM`-th 𝐴.
    - `B` (N, *Nd, xyz), `B[:,iM,:]` is the `iM`-th 𝐵.
    """
    shape = Beff.shape

    # defaults
    device, dtype = Beff.device, Beff.dtype
    dkw = {'device': device, 'dtype': dtype}
    T1 = tensor([[inf]], **dkw) if (T1 is None) else T1.to(device)
    T2 = tensor([[inf]], **dkw) if (T2 is None) else T2.to(device)
    γ = tensor([[γH]], **dkw) if (γ is None) else γ.to(device)
    dt = tensor([[dt0]], **dkw) if (dt0 is None) else dt.to(device)

    # reshaping
    NNd, nT = shape[0:-2], shape[-1]
    T1, T2, γ = map(lambda x: x.expand(NNd), (T1, T2, γ))

    # C/Python `reshape/view` is different from Fortran/MatLab/Julia `reshape`
    s1, s0 = NNd+(1, 1), NNd+(1, 4)

    kw = {'device': Beff.device, 'dtype': Beff.dtype}

    AB = torch.cat([torch.ones(s1, **kw), torch.zeros(s0, **kw),
                    torch.ones(s1, **kw), torch.zeros(s0, **kw),
                    torch.ones(s1, **kw), torch.zeros(s1, **kw)],
                   dim=-1).view(NNd+(3, 4))  # -> (N, *Nd, xyz, 3+1)

    E1, E2 = (torch.exp(-dt/T1)[..., None],        # (N, 1, 1)
              torch.exp(-dt/T2)[..., None, None])  # (N, 1, 1, 1)
    E1_1 = E1.squeeze(dim=-1) - 1
    γ2πdt = 2*π*γ*dt  # Hz/Gauss -> Rad/Gauss

    # simulation
    for t in range(nT):
        u, ϕ = beff2uϕ(Beff[..., t], γ2πdt)

        if torch.any(ϕ != 0):
            AB1 = uϕrot(u, ϕ, AB)
        else:
            AB1 = AB

        # Relaxation
        AB1[..., 0:2, :] *= E2
        AB1[..., 2, :] *= E1
        AB1[..., 2, 3] -= E1_1
        AB, AB1 = AB1, AB

    A, B = AB[..., 0:3], AB[..., 3]

    return A, B


def blochsim_1step(
        M: torch.Tensor, M1: torch.Tensor, b: torch.Tensor,
        E1: torch.Tensor, E1_1: torch.Tensor, E2: torch.Tensor,
        γ2πdt: torch.Tensor):
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
    u, ϕ = beff2uϕ(b, γ2πdt)

    if torch.any(ϕ != 0):
        M1 = uϕrot(u, ϕ, M)
    else:
        M1 = M
    # Relaxation
    M1[..., 0:2] *= E2[..., None]
    M1[..., 2] *= E1
    M1[..., 2] -= E1_1

    M, M1 = M1, M
    return M, M1


def blochsim(
        M: torch.Tensor, Beff: torch.Tensor,
        T1: torch.Tensor = None, T2: torch.Tensor = None,
        γ: torch.Tensor = None, dt: torch.Tensor = None):
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

    # defaults and move to the same device
    device, dtype = M.device, M.dtype
    Beff = Beff.to(device)
    dkw = {'device': device, 'dtype': dtype}
    T1 = tensor([[inf]], **dkw) if (T1 is None) else T1.to(device)
    T2 = tensor([[inf]], **dkw) if (T2 is None) else T2.to(device)
    γ = tensor([[γH]], **dkw) if (γ is None) else γ.to(device)
    dt = tensor([[dt0]], **dkw) if (dt0 is None) else dt.to(device)

    # reshaping
    d = M.dim()-1
    T1, T2, γ = map(lambda x: x.reshape(x.shape+(d-x.dim())*(1,)), (T1, T2, γ))

    E1, E2 = torch.exp(-dt/T1), torch.exp(-dt/T2)[..., None]
    E1_1 = E1 - 1
    γ2πdt = 2*π*γ*dt  # Hz/Gauss -> Rad/Gauss

    # simulation
    for t in range(Beff.shape[-1]):
        u, ϕ = beff2uϕ(Beff[..., t], γ2πdt)
        if torch.any(ϕ != 0):
            M1 = uϕrot(u, ϕ, M)
        else:
            M1 = M
        # Relaxation
        M1[..., 0:2] *= E2
        M1[..., 2] *= E1
        M1[..., 2] -= E1_1

        M, M1 = M1, M

    return M


def blochsim_ab(M: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
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
