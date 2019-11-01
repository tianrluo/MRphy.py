import torch
import torch.nn.functional as F
from torch import tensor
from math import pi as π, inf

"""
*General Comments*:
- `N`  for batch size
- `nM` for nSpins
- `nT` for the number of time points
- `xy` basically means that dimension has length of 2
- `xyz` means that dimension has length of 3
"""


def rfgr2B(
        rf, gr,
        loc=tensor([0, 0, 0])[None, None, ...],
        Δf=None,
        b1Map=tensor([1, 0])[None, None, ...],
        γ=tensor(4257.6)):
    """
        B = rfgr2B(rf, gr, loc, Δf, b1Map, γ)
    *INPUTS*:
    - `rf` (N,xy, nT,(nCoils)) "Gauss", `xy` for separating real and imag part.
    - `gr` (N,xyz,nT) "Gauss/cm"
    *OPTIONALS*:
    - `loc`(N,nM,xyz) "cm", locations.
    - `Δf` (N,nM,) "Hz", off-resonance.
    - `b1Map` (N,nM,xy,(nCoils)) a.u., , transmit sensitivity.
    - `γ`(1,) "Hz/Gauss", gyro-ratio
    *OUTPUTS*:
    - `B`  (N,nM,xyz,nT)
    """
    Bz = (loc @ gr)[:, :, None, :]  # (N, nM, nT) -> (N, nM, 1, nT)
    if Δf is not None:
        Bz += Δf[..., None, None] / γ  # Δf: (N, nM) -> (N, nM, 1, 1)

    if rf.dim() == 3:
        rf = rf[:, None, ..., None]  # (N,xy,nT)->(N,1,xy,nT,1)
    elif rf.dim() == 4:
        rf = rf[:, None, ...]  # (N,xy,nT,nCoils)->(N,1,xy,nT,nCoils)

    if b1Map.dim() == 3:
        b1Map = b1Map[..., None, None]  # (N,nM,xy)->(N,nM,xy,1,1)
    elif b1Map.dim() == 4:
        b1Map = b1Map[..., None, :]  # (N,nM,xy,nCoils)->(N,nM,xy,1,nCoils)

    # Real as `Bx`, Imag as `By`.
    Bx = torch.sum(b1Map[..., 0:1, :, :]*rf[:, :, 0:1, ...]
                   - b1Map[..., 1:2, :, :]*rf[:, :, 1:2, ...],
                   dim=4).expand_as(Bz)  # -> (N, nM, x, nT)
    By = torch.sum(b1Map[..., 0:1, :, :]*rf[:, :, 1:2, ...]
                   + b1Map[..., 1:2, :, :]*rf[:, :, 0:1, ...],
                   dim=4).expand_as(Bz)  # -> (N, nM, y, nT)

    B = torch.cat([Bx, By, Bz], dim=2)  # -> (N, nM, xyz, nT)
    return B


def B2UΦ(B, γ2πdt):
    """
        U, Φ = B2UΦ(B, γ2πdt)
    *INPUTS*:
    - `B` (N, nM, xyz) "Gauss", B-effective, magnetic field applied on `M`.
    - `γ2πdt` (1,) "Rad/Gauss", gyro ratio in radians, global.
    *OUTPUTS*:
    - `U` (N, nM, xyz), rotation axis
    - `Φ` (N, nM, xyz), rotation angle
    """
    U = F.normalize(B, dim=2)
    Φ = -torch.norm(B, dim=2) * γ2πdt  # negate: BxM -> MxB
    return U, Φ


def UΦRot(U, Φ, Vi):
    """
        Vo = UΦRot(U, Φ, Vi)
    Apply axis-angle, `U-Phi` rotation on `V`. Rotation is broadcasted on `V`.
    <en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle>

    *INPUTS*:
    - `U`  (N, nM, xyz), 3D rotation axes, assumed unitary;
    - `Φ`  (N, nM,), rotation angles;
    - `Vi` (N, nM, xyz, ...), vectors to be rotated;
    *OUTPUTS*:
    - `Vo` (N, nM, xyz, ...), vectors rotated;
    """
    Φ = Φ[..., None]
    if Vi.dim() == 4:
        Φ, U = Φ[..., None], U[..., None]  # (N, nM, xyz) -> (N, nM, xyz, 1)

    cΦ, sΦ = torch.cos(Φ), torch.sin(Φ)  # ^OPT: maybe in-place to avoid alloc

    Vo = (cΦ*Vi + (1-cΦ)*torch.sum(U*Vi, 2, keepdim=True)*U 
          + sΦ*torch.cross(U, Vi, dim=2))

    return Vo


def blochSim_1Step(M, M1, b, E1, E1_1, E2, γ2πdt):
    """
        blochSim_1Step(M, M1, b, E1, E1_1, E2, γ2πdt)
    *INPUTS*:
    - M (N, nM, xyz), Magnetic spins, assumed equilibrium magnitude `[0 0 1]`.
    - M1 (N, nM, xyz), pre-allocated variable for `UΦRot` output.
    - b (N, nM, xyz) "Gauss", B-effective, magnetic field applied.
    - E1 (N, 1,) a.u., T1 reciprocal exponential, global.
    - E1_1 (N, 1,) a.u., T1 reciprocal exponential subtracted by `1`, global.
    - E2 (N, 1,) a.u., T2 reciprocal exponential, global.
    - γ2πdt (N, 1,) "rad/Gauss", gyro ratio mutiplied by `dt`, global.
    *OUTPUTS*:
    - M (nM, xyz), Magetic spins after simulation.
    *Notes*:
      spin history during simulations is not provided at the moment.
    """
    u, ϕ = B2UΦ(b, γ2πdt)
    if torch.any(ϕ != 0):
        M1 = UΦRot(u, ϕ, M)
    # Relaxation
    M1[:, :, 0:2] *= E2
    M1[:, :, 2] *= E1
    M1[:, :, 2] -= E1_1

    M, M1 = M1, M
    return M, M1


def blochSim(
        M, B,
        T1=tensor(inf),
        T2=tensor(inf),
        γ=tensor(4257.6),
        dt=tensor(4e-6)):
    """
    *INPUTS*:
    - M (N, nM, xyz), Magnetic spins, assumed equilibrium magnitude `[0 0 1]`.
    - B (N, nM, xyz, nT) "Gauss", B-effective, magnetic field applied.
    *OPTIONALS*:
    - T1 (N, 1,) "Sec", T1 relaxation, global.
    - T2 (N, 1,) "Sec", T2 relaxation, global.
    - γ (N, 1,) "Hz/Gauss", gyro ratio in Hertz, global.
    - dt (N, 1,) "Sec", dwell time, global.
    *OUTPUTS*:
    - M (nM, xyz), Magetic spins after simulation.
    *Notes*:
      spin history during simulations is not provided at the moment.
    """
    E1, E2 = torch.exp(-dt/T1), torch.exp(-dt/T2)
    E1_1 = E1 - 1
    γ2πdt = 2*π*γ*dt  # Hz/Gauss -> Rad/Gauss

    M1 = M.clone()

    for t in range(B.shape[-1]):
        u, ϕ = B2UΦ(B[..., t], γ2πdt)
        if torch.any(ϕ != 0):
            M1 = UΦRot(u, ϕ, M)
        # Relaxation
        M1[:, :, 0:2] *= E2
        M1[:, :, 2] *= E1
        M1[:, :, 2] -= E1_1

        M, M1 = M1, M

    return M
