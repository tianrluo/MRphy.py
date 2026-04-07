r"""Steady-state simulations for SPGR-based sequences."""
import math
from typing import Optional, Union

import torch
from torch import Tensor

from mrphy.mobjs import SpinArray, SpinCube, Pulse

__all__ = ['spgr_ovs', 'target_ovs']


def spgr_ovs(
    spin: Union[SpinArray,SpinCube],
    pulse: Pulse,
    *,
    doEmbed: bool = False, 
    doRelax: bool = True, 
    doUpdate: bool = False,
    betaoff: bool = False,
    loc: Optional[Tensor] = None, 
    loc_: Optional[Tensor] = None,
    Δf: Optional[Tensor] = None, 
    Δf_: Optional[Tensor] = None,
    b1Map: Optional[Tensor] = None, 
    b1Map_: Optional[Tensor] = None,
    alpha: float = 0, 
    TR: float = 55e-3
) -> Tensor:
    r"""Compute the steady-state magnetization for an outer-volume-suppression (OVS) SPGR sequence.

    Assumes the following sequence structure:
        ``[beta – alpha – readout] × N_rep``
    where ``beta`` (the input ``pulse``) is a saturation preparation pulse and
    ``alpha`` is a non-selective excitation. If ``alpha=0``, the pulse is
    treated as a selective excitation rather than a saturation preparation.

    The output is the steady-state magnetization right before the ``alpha``
    pulse.

    Usage:
        ``Mss_ = mrphy.steady_state.spgr_ovs(spin, pulse, *, alpha, TR, ...)``

    Inputs:
        - ``spin``: mrphy.mobjs.SpinArray or mrphy.mobjs.SpinCube.
        - ``pulse``: mrphy.mobjs.Pulse.
    Optionals:
        - ``loc`` ⊻ ``loc_``: `(N,*Nd ⊻ nM,xyz)`, "cm", locations. \
          Ignored when ``spin`` is a SpinCube (uses cube's intrinsic loc).
        - ``Δf`` ⊻ ``Δf_``: `(N,*Nd ⊻ nM)`, "Hz", off-resonance. \
          Ignored when ``spin`` is a SpinCube (uses cube's intrinsic Δf).
        - ``b1Map`` ⊻ ``b1Map_``: `(N,*Nd ⊻ nM,xy,(nCoils))`, transmit sensitivity.
        - ``doEmbed``: [t/F], return ``M`` or ``M_``.
        - ``doRelax``: [T/f], do relaxation during Bloch simulation.
        - ``doUpdate``: [t/F], update ``spin.M_`` to the SS magnetization.
        - ``betaoff``: [t/F], treat the beta pulse as zero flip angle.
        - ``alpha``: float, flip angle of the non-selective excitation, in [deg].
        - ``TR``: float, repetition time, in [sec].
    Outputs:
        - ``Mss`` ⊻ ``Mss_``: `(N,*Nd ⊻ nM,xyz)`
    """

    if isinstance(spin, SpinCube):
        assert (b1Map_ is None) or (b1Map is None)
        b1Map_ = (b1Map_ if b1Map is None else spin.extract(b1Map))
        MT_ = spin.spinarray.applypulse(
            pulse, doEmbed=False, doRelax=doRelax, doUpdate=False,
            loc_=spin.loc_, Δf_=spin.Δf_, b1Map_=b1Map_
        )
        sa = spin.spinarray
    else:
        MT_ = spin.applypulse(
            pulse, doEmbed=False, doRelax=doRelax, doUpdate=False,
            loc=loc, loc_=loc_, Δf=Δf, Δf_=Δf_, b1Map=b1Map, b1Map_=b1Map_
        )
        sa = spin

    MT_ = torch.nan_to_num(MT_)  # (N, nM, xyz)

    if betaoff:
        Beta_ = torch.zeros_like(MT_[..., 0])
    else:
        Beta_ = torch.arctan(
            (MT_[..., 0]**2 + MT_[..., 1]**2).sqrt()
            / (MT_[..., 2] + 1e-8)
        )  # saturation angle in rad, (N, nM)

    T_r = TR - (pulse.dt * pulse.rf.shape[-1]).item()  # time remaining after beta to end of TR
    E1_ = torch.exp(-T_r / sa.T1_)  # (N, nM)

    denom = 1 - torch.cos(Beta_) * E1_ * math.cos(math.radians(alpha))
    scale = sa.M_[..., 2] * (1 - E1_) / denom

    Mss_ = torch.zeros_like(MT_)
    Mss_[..., 2] = scale * torch.cos(Beta_)  # steady-state Mz before alpha
    Mss_[..., 0] = scale * torch.sin(Beta_)  # steady-state Mxy before alpha (along x)

    if doUpdate:
        sa.M_ = Mss_

    Mss_ = (spin.embed(Mss_) if doEmbed else Mss_)
    return Mss_


def target_ovs(
    cube: SpinCube,
    beta_iv: float, 
    beta_ov: float, 
    iv: Tensor, 
    ov: Tensor,
    *,
    weight_iv: float = 1.0, 
    weight_ov: float = 1.0,
    doEmbed: bool = True,
    alpha: float = 0, 
    TR: float = 55e-3
) -> tuple[Tensor, Tensor]:
    r"""Compute the target steady-state magnetization profile for an OVS SPGR sequence.

    Assumes the following sequence structure:
        ``[beta – alpha – readout] × N_rep``
    The target is the desired steady-state magnetization right before ``alpha``,
    specified via target beta flip angles for inner- and outer-volume regions.

    Usage:
        ``d, weight = mrphy.steady_state.target_ovs(cube, 0, 90, iv, ov, alpha=20, TR=80e-3)``

    Inputs:
        - ``cube``: mrphy.mobjs.SpinCube.
        - ``beta_iv``: target inner-volume beta flip angle, in [deg].
        - ``beta_ov``: target outer-volume beta flip angle, in [deg].
        - ``iv``: inner-volume boolean mask, `(N, *nM)`.
        - ``ov``: outer-volume boolean mask, `(N, *nM)`.
    Optionals:
        - ``weight_iv``: loss weighting for the iv region.
        - ``weight_ov``: loss weighting for the ov region.
        - ``doEmbed``: [T/f], return embedded or compact form.
        - ``alpha``: float, flip angle of the non-selective excitation, in [deg].
        - ``TR``: float, repetition time, in [sec].
    Outputs:
        - ``d``: target SS magnetization, `(N, *nM, xyz)`.
        - ``weight``: region weights, `(N, *nM)`.
    """
    d = torch.zeros(iv.shape + (3,), device=cube.device)
    M0 = cube.M[..., 2]  # (N, *nM)

    beta = beta_iv * iv + beta_ov * ov  # (N, *nM), flip angle map in deg
    E1 = torch.exp(-TR / cube.T1)       # (N, *nM)

    denom = 1 - torch.cos(torch.deg2rad(beta)) * math.cos(math.radians(alpha)) * E1
    scale = M0 * (1 - E1) / denom
    d[..., 2] = scale * torch.cos(torch.deg2rad(beta))  # Mz before alpha
    d[..., 0] = scale * torch.sin(torch.deg2rad(beta))  # Mxy before alpha (along x)
    d = d.nan_to_num()

    weight = weight_iv * iv + weight_ov * ov  # (N, *nM)

    d = d if doEmbed else cube.extract(d)
    weight = weight if doEmbed else cube.extract(weight)

    return d, weight
