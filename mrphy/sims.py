r"""Simulation codes with explicit Jacobian operations.
"""

from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import _ContextMethodMixin as CTX

from mrphy import γH, dt0, π


# TODO:
# - Avoid caching when needs_input_grad is False
# - Create `BlochSim_rfgr` that directly computes grads w.r.t. `rf` and `gr`.


__all__ = ['blochsim']

_contiguous_format = torch.contiguous_format


class BlochSim(Function):
    r"""BlochSim with explict Jacobian operation (backward)

    This operator is only differentiable w.r.t. ``Mi`` and ``Beff``.

    """

    @staticmethod
    def forward(
        ctx: CTX,
        Mi: Tensor,
        Beff: Tensor,
        T1: Optional[Tensor],
        T2: Optional[Tensor],
        γ: Tensor,
        dt: Tensor
    ) -> Tensor:
        r"""Forward evolution of Bloch simulation

        Inputs:
            - ``ctx``: `(1,)`, pytorch CTX cacheing object
            - ``Mi``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
              [0 0 1]
            - ``Beff``: `(N, *Nd, nT, xyz)`, "Gauss", B-effective, magnetic \
              field.
            - ``T1``: `(N ⊻ 1, *Nd ⊻ len(Nd)*(1,), 1, 1)`, "Sec", T₁
            - ``T2``: `(N ⊻ 1, *Nd ⊻ len(Nd)*(1,), 1, 1)`, "Sec", T₂
            - ``γ``:  `(N ⊻ 1, *Nd ⊻ len(Nd)*(1,), 1, 1)`, "Hz/Gauss", gyro.
            - ``dt``: `(N ⊻ 1, len(Nd)*(1,), 1, 1)`, "Sec", dwell time.
        Outputs:
            - ``Mo``: `(N, *Nd, xyz)`, Magetic spins after simulation.
        """
        NNd, nT = Beff.shape[:-2], Beff.shape[-2]
        # (t)ensor (k)ey(w)ord, contiguous to avoid alloc/copy when reshape
        tkw = {'memory_format': _contiguous_format,
               'dtype': Mi.dtype, 'device': Mi.device}

        # %% Preprocessing
        γ2πdt = 2*π*γ*dt
        γBeff = torch.empty(Beff.shape, **tkw)
        torch.mul(γ2πdt, Beff, out=γBeff)
        Mi = Mi.clone(memory_format=_contiguous_format)[..., None, :]
        # γBeff = γ2πdt*Beff.contiguous()

        assert((T1 is None) == (T2 is None))  # both or neither

        if T1 is None:  # relaxations ignored
            E = e1_1 = None
            fn_relax_ = lambda m1: None
        else:
            E1, E2 = -dt/T1, -dt/T2
            E1.exp_(), E2.exp_()  # should have fewer alloc than exp(-dt/T1)
            E, e1_1 = torch.cat((E2, E2, E1), dim=-1), E1-1
            fn_relax_ = lambda m1: (m1.mul_(E))[..., 2:3].sub_(e1_1)

        # Pre-allocate intermediate variables, in case of overhead alloc's
        v = torch.empty(Mi.shape, **tkw)  # (N, *Nd, xyz, 1)

        # %% Other variables to be cached
        m0 = Mi
        Mhst = torch.empty(NNd+(nT, 3), **tkw)
        Φ = torch.empty(NNd+(nT, 1), **tkw)
        cΦ_1 = torch.empty(NNd+(nT, 1), **tkw)
        sΦ = torch.empty(NNd+(nT, 1), **tkw)
        UtM0 = torch.empty(NNd+(nT, 1), **tkw)

        # %% Simulation. could we learn to live right.
        for m1, γbeff, ϕ, cϕ_1, sϕ, utm0 in zip(
            Mhst.split(1, dim=-2),
            γBeff.split(1, dim=-2),
            Φ.split(1, dim=-2),
            cΦ_1.split(1, dim=-2),
            sΦ.split(1, dim=-2),
            UtM0.split(1, dim=-2),
        ):
            # Rotation
            torch.norm(γbeff, dim=-1, keepdim=True, out=ϕ)
            ϕ.clamp_(min=1e-12)
            torch.div(γbeff, ϕ, out=γbeff)
            u = γbeff

            torch.sin(ϕ, out=sϕ)
            torch.cos(ϕ, out=cϕ_1)
            cϕ_1.sub_(1)  # (cϕ-1)

            # wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
            # Angle is `-ϕ` as Bloch-eq is 𝑀×𝐵
            # m₁ = R(u, -ϕ)m₀ = cϕ*m₀ + (1-cϕ)*uᵀm₀*u - sϕ*u×m₀
            # m₁ = m₀ - sϕ*u×m₀ + (cϕ-1)*(m₀ - uᵀm₀*u), in-place friendly
            torch.mul(u, m0, out=m1)  # using m₁ as an temporary storage
            torch.sum(m1, dim=-1, keepdim=True, out=utm0)

            torch.cross(u, m0, dim=-1, out=m1)  # u×m₀
            torch.addcmul(m0, sϕ, m1, value=-1, out=m1)  # m₀ - sϕ*(u×m₀)

            torch.addcmul(m0, utm0, u, value=-1, out=v)  # m₀ - uᵀm₀*u

            torch.addcmul(m1, cϕ_1, v, out=m1)  # m₀-sϕ*u×m₀+(cϕ-1)*(m₀-uᵀm₀*u)

            # Relaxation
            fn_relax_(m1)

            m0 = m1

        ctx.save_for_backward(
            Mi, Mhst, γBeff, Φ, cΦ_1, sΦ, UtM0, E, e1_1, γ2πdt
        )
        Mo = Mhst[..., -1, :].clone()  # -> (N, *Nd, xyz)
        return Mo

    @staticmethod
    def backward(
        ctx: CTX,
        grad_Mo: Tensor
    ) -> Tuple[Tensor, Tensor, None, None, None, None]:
        r"""Backward evolution of Bloch simulation Jacobians

        Inputs:
            - ``ctx``: `(1,)`, pytorch CTX cacheing object
            - ``grad_Mo``: `(N, *Nd, xyz)`, derivative w.r.t. output Magetic \
              spins.
        Outputs:
            - ``grad_Mi``: `(N, *Nd, xyz)`, derivative w.r.t. input Magetic \
              spins.
            - ``grad_Beff``: `(N,*Nd,nT,xyz)`, derivative w.r.t. B-effective.
            - None*4, this implemendation do not provide derivatives w.r.t.: \
              `T1`, `T2`, `γ`, and `dt`.
        """
        # grads of configuration variables are not supported yet
        needs_grad = ctx.needs_input_grad
        grad_Beff = grad_Mi = grad_T1 = grad_T2 = grad_γ = grad_dt = None

        if not any(needs_grad[0:2]):  # (Mi,Beff;T1,T2,γ,dt):
            return grad_Mi, grad_Beff, grad_T1, grad_T2, grad_γ, grad_dt

        # %% Jacobians. If we turn back time,
        # ctx.save_for_backward(Mi, Mhst, γBeff, E, e1_1, γ2πdt)
        Mi, Mhst, γBeff, Φ, cΦ_1, sΦ, UtM0, E, e1_1, γ2πdt = ctx.saved_tensors
        NNd = γBeff.shape[:-2]
        # (t)ensor (k)ey(w)ord, contiguous to avoid alloc/copy when reshape
        tkw = {'memory_format': _contiguous_format,
               'dtype': Mi.dtype, 'device': Mi.device}

        # assert((E is None) == (e1_1 is None))  # both or neither
        if E is None:  # relaxations ignored
            fn_relax_h1_ = lambda h1: None
            fn_relax_m1_ = lambda m1: None
        else:
            fn_relax_h1_ = lambda h1: h1.mul_(E)

            def fn_relax_m1_(m1):
                m1[..., :, 2:3].add_(e1_1)
                m1.div_(E)
                return

        # Pre-allocate intermediate variables, in case of overhead alloc's
        h0 = torch.empty(NNd+(1, 3), **tkw)

        u, uxh1 = (torch.empty(NNd+(1, 3), **tkw) for _ in range(2))
        ϕ, cϕ_1, sϕ, uth1 = (torch.empty(NNd+(1, 1), **tkw) for _ in range(4))
        # ϕis0 = torch.empty(NNd+(1, 1),
        #                    memory_format=tkw['memory_format'],
        #                    device=tkw['device'], dtype=torch.bool)
        h1 = grad_Mo.clone(memory_format=_contiguous_format)[..., None, :]
        # u_dflt = torch.tensor([[0.], [0.], [1.]],  # (xyz, 1)
        #                       device=tkw['device'], dtype=tkw['dtype'])

        m1 = Mhst.narrow(-2, -1, 1)

        # scale by -γ2πdt, so output ∂L/∂B no longer needs multiply by -γ2πdt
        h1.mul_(-γ2πdt)
        for m0, γbeff, ϕ, cϕ_1, sϕ, utm0 in zip(
            reversed((Mi,)+Mhst.split(1, dim=-2)[:-1]),
            reversed(γBeff.split(1, dim=-2)),
            reversed(Φ.split(1, dim=-2)),
            reversed(cΦ_1.split(1, dim=-2)),
            reversed(sΦ.split(1, dim=-2)),
            reversed(UtM0.split(1, dim=-2)),
        ):
            # %% Ajoint Relaxation:
            fn_relax_h1_(h1)  # h₁ → h̃₁ ≔ ∂L/∂m̃₁ = E∂L/∂m₁
            fn_relax_m1_(m1)  # m₁ → m̃₁ ≔ Rm₀ = E⁻¹m₁

            # %% Adjoint Rotations:
            u.copy_(γbeff)

            # TODO: Resolve singularities of ϕ=0, control pov?
            # torch.logical_not(ϕ, out=ϕis0)
            # if torch.any(ϕis0):
            #     u[ϕis0[..., 0, 0]] = u_dflt

            # %% Assemble h₀: (R(u, -ϕ)ᵀ ≡ R(u, ϕ))
            # h₀ ≔ R(u, ϕ)h̃₁ = cϕ*h₁ + (1-cϕ)*uᵀh₁*u + sϕ*u×h₁
            # h₀ = h̃₁ + (cϕ-1)*(h̃₁ - uᵀh̃₁*u) + sϕ*u×h̃₁, in-place friendly
            torch.mul(u, h1, out=h0)  # using h0 as an temporary storage
            torch.sum(h0, dim=-1, keepdim=True, out=uth1)  # uᵀh̃₁

            torch.cross(u, h1, dim=-1, out=uxh1)  # u×h̃₁
            torch.addcmul(h1, uth1, u, value=-1, out=h0)  # h̃₁-uᵀh̃₁*u

            torch.addcmul(h1, cϕ_1, h0, out=h0)  # h̃₁ + (cϕ-1)*(h̃₁-uᵀh̃₁*u)

            # Finish: h₀ = h̃₁ + (cϕ-1)*(h̃₁ - uᵀh̃₁*u) + sϕ*u×h̃₁
            torch.addcmul(h0, sϕ, uxh1, value=1, out=h0)

            # %% Assemble ∂L/∂B[..., t], store into γbeff
            # -γδt⋅(+sϕ/ϕ⋅m₀×h̃₁
            #       +(cϕ-1)/ϕ⋅(uᵀm₀⋅h̃₁+uᵀh̃₁⋅m₀)
            #       +((sϕ/ϕ⋅m₀-m̃₁)ᵀ(u×h̃₁)-2(cϕ-1)/ϕ⋅uᵀm₀⋅uᵀh̃1)*u )

            cϕ_1.div_(ϕ), sϕ.div_(ϕ)  # cϕ-1, sϕ → (cϕ-1)/ϕ, sϕ/ϕ
            # if torch.any(ϕis0):  # handle division-by-0
            #     cϕ_1[ϕis0], sϕ[ϕis0] = 0, 1

            # %%% sϕ/ϕ⋅(m₀×h̃₁)
            torch.cross(m0, h1, dim=-1, out=γbeff)  # m₀×h̃₁
            γbeff.mul_(sϕ)  # sϕ/ϕ⋅(m₀×h̃₁)

            # %%% sϕ/ϕ⋅(m₀×h̃₁) + (cϕ-1)/ϕ⋅(uᵀm₀⋅h̃₁+uᵀh̃₁⋅m₀)
            h1.mul_(utm0)  # uᵀm₀⋅h̃₁
            torch.addcmul(h1, uth1, m0, out=h1)  # (uᵀm₀⋅h̃₁+uᵀh̃₁⋅m₀)
            torch.addcmul(γbeff, cϕ_1, h1, value=1, out=γbeff)

            # %%% sϕ/ϕ⋅(m₀×h̃₁) + (cϕ-1)/ϕ⋅(uᵀm₀⋅h̃₁+uᵀh̃₁⋅m₀)
            #     -((m̃₁-sϕ/ϕ⋅m₀)ᵀ(u×h̃₁) + 2(cϕ-1)/ϕ⋅uᵀh̃1⋅uᵀm₀)⋅u

            # (m̃₁-sϕ/ϕ⋅m₀)ᵀ(u×h̃₁)
            torch.addcmul(m1, sϕ, m0, value=-1, out=m1)  # (m̃₁-sϕ/ϕ⋅m₀)
            m1.mul_(uxh1)
            torch.sum(m1, dim=-1, keepdim=True, out=sϕ)

            # ((m̃₁-sϕ/ϕ⋅m₀)ᵀ(u×h̃₁) + 2(cϕ-1)/ϕ⋅uᵀh̃₁⋅uᵀm₀)
            uth1.mul_(utm0)  # uᵀh̃1⋅uᵀm₀
            torch.addcmul(sϕ, cϕ_1, uth1, value=2, out=cϕ_1)

            torch.addcmul(γbeff, cϕ_1, u, value=-1, out=γbeff)

            m1, h1, h0 = m0, h0, h1

        # %% Clean up
        grad_Beff = γBeff

        # undo the multiply by -γ2πdt on h1
        grad_Mi = h1[..., 0, :].div_(-γ2πdt[0, ...]) if needs_grad[0] else None
        # forward(ctx, Mi, Beff; T1, T2, γ, dt):
        return grad_Mi, grad_Beff, grad_T1, grad_T2, grad_γ, grad_dt


def blochsim(
    Mi: Tensor, Beff: Tensor, *,
    T1: Optional[Tensor] = None, T2: Optional[Tensor] = None,
    γ: Tensor = γH, dt: Tensor = dt0
) -> Tensor:
    r"""Bloch simulator with explicit Jacobian operation.

    This function is only differentiable w.r.t. ``Mi`` and ``Beff``.

    Setting `T1=T2=None` to opt for simulation ignoring relaxation.

    Usage:
        ``Mo = blochsim(Mi, Beff, *, T1, T2, γ, dt)``
        ``Mo = blochsim(Mi, Beff, *, T1=None, T2=None, γ, dt)``
    Inputs:
        - ``Mi``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
          [[[0 0 1]]].
        - ``Beff``: `(N, *Nd, nT, xyz)`, "Gauss", B-effective, magnetic field.
    Optionals:
        - ``T1``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T1 relaxation.
        - ``T2``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T2 relaxation.
        - ``γ``:  `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Hz/Gauss", gyro ratio.
        - ``dt``: `()` ⊻ `(N ⊻ 1,)`, "Sec", dwell time.
    Outputs:
        - ``Mo``: `(N, *Nd, xyz)`, Magetic spins after simulation.

    .. tip::
        For alternative implementation:
        Storing history for `U`, `Φ` and `UtM0` etc., which are also used in
        `backward`, may avoid redundant computation, but comsumes more RAM.
    """

    # %% Defaults and move to the same device
    assert(Mi.shape[:-1] == Beff.shape[:-2])
    Beff, ndim = Beff.to(Mi.device), Beff.ndim

    # Make {γ, dt, T1, T2} compatible with (N, *Nd, :, :)
    γ, dt = (x.reshape(x.shape+(ndim-x.ndim)*(1,)) for x in (γ, dt))

    assert((T1 is None) == (T2 is None))  # both or neither
    if T1 is not None:
        T1, T2 = (x.reshape(x.shape+(ndim-x.ndim)*(1,)) for x in (T1, T2))

    return BlochSim.apply(Mi, Beff, T1, T2, γ, dt)


class FreePrec(Function):
    r"""Free precession with explicit Jacobian operation (backward)

    This operator is only differentiable w.r.t. ``Mi``.

    """

    @staticmethod
    def forward(
        ctx: CTX, Mi: Tensor, dur: Tensor,
        T1: Optional[Tensor], T2: Optional[Tensor], Δf: Optional[Tensor]
    ) -> Tensor:
        r"""Forward operation of free precession

        Inputs:
            - ``ctx``: `(1,)`, pytorch CTX cacheing object
            - ``Mi``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
              [0 0 1]
            - ``dur``: `(N ⊻ 1, len(Nd)*(1,), 1)`, "Sec", dwell time.
            - ``T1``: `(N ⊻ 1, *Nd ⊻ len(Nd)*(1,), 1)`, "Sec", T₁
            - ``T2``: `(N ⊻ 1, *Nd ⊻ len(Nd)*(1,), 1)`, "Sec", T₂
            - ``Δf``: `(N ⊻ 1, *Nd ⊻ len(Nd)*(1,))`, "Sec", T₂
        Outputs:
            - ``Mo``: `(N, *Nd, xyz)`, Magetic spins after simulation.
        """  # could we learn to live right.

        Mo = Mi.clone(memory_format=_contiguous_format)

        # Precession
        cϕ = sϕ = tmp = None
        if Δf is not None:  # positive Δf dephases clock-wise/negatively
            sϕ = -(2*π)*Δf*dur[..., 0]
            cϕ = torch.cos(sϕ)
            sϕ.sin_()  # ϕ is now sϕ

            tmp = Mo[..., 0].clone(memory_format=_contiguous_format)  # Mix
            Mo[..., 0].mul_(cϕ)  # cϕ*Mix
            torch.addcmul(Mo[..., 0], sϕ, Mo[..., 1], value=-1,
                          out=Mo[..., 0])  # Mox = cϕ*Mix - sϕ*Miy

            Mo[..., 1].mul_(cϕ)
            torch.addcmul(Mo[..., 1], sϕ, tmp,
                          out=Mo[..., 1])  # Moy = sϕ*Mix + cϕ*Miy

        # Relaxation
        E1 = E2 = E1_1 = None
        assert((T1 is None) == (T2 is None))  # both or neither

        if T1 is not None:
            E1, E2 = -dur/T1, -dur/T2
            E1_1 = torch.expm1(E1)  # E1 - 1
            E1.exp_(), E2.exp_()  # should have fewer alloc than exp(-dt/T1)
            Mo[..., 0:2].mul_(E2)
            Mo[..., 2:3].mul_(E1).sub_(E1_1)

        ctx.save_for_backward(cϕ, sϕ, E1, E2, tmp)

        return Mo

    @staticmethod
    def backward(
        ctx: CTX, grad_Mo: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        r"""Backward operation of free precession

        Inputs:
            - ``ctx``: `(1,)`, pytorch CTX cacheing object
            - ``grad_Mo``: `(N, *Nd, xyz)`, derivative w.r.t. output Magetic \
              spins.
        Outputs:
            - ``grad_Mi``: `(N, *Nd, xyz)`, derivative w.r.t. input Magetic \
              spins.
            - None*4, this implemendation do not provide derivatives w.r.t.: \
              `dur`, `T1`, `T2`, and `Δf`.
        """  # If we turn back time,
        # grads of configuration variables are not supported yet
        needs_grad = ctx.needs_input_grad
        grad_Mi = grad_dur = grad_T1 = grad_T2 = grad_Δf = None

        if not any(needs_grad[0:1]):
            return grad_Mi, grad_dur, grad_T1, grad_T2, grad_Δf

        grad_Mi = grad_Mo.clone(memory_format=_contiguous_format)

        # ctx.save_for_backward(cϕ, sϕ, E1, E2, E1_1)
        cϕ, sϕ, E1, E2, tmp = ctx.saved_tensors

        # Relaxation
        if E1 is not None:
            grad_Mi[..., 0:2].mul_(E2)
            grad_Mi[..., 2:3].mul_(E1)

        # Precession
        if tmp is not None:
            tmp.copy_(grad_Mi[..., 1])  # copy of ∂Moy
            grad_Mi[..., 1].mul_(cϕ)  # cϕ*∂Moy
            torch.addcmul(grad_Mi[..., 1], sϕ, grad_Mi[..., 0], value=-1,
                          out=grad_Mi[..., 1])  # ∂Miy = -sϕ∂Mox + cϕ*∂Moy

            grad_Mi[..., 0].mul_(cϕ)  # cϕ*∂Mox
            torch.addcmul(grad_Mi[..., 0], sϕ, tmp,
                          out=grad_Mi[..., 0])  # ∂Mix = cϕ*∂Mox + sϕ∂Moy

        return grad_Mi, grad_dur, grad_T1, grad_T2, grad_Δf


def freeprec(
    Mi: Tensor, dur: Tensor, *,
    T1: Optional[Tensor] = None, T2: Optional[Tensor] = None,
    Δf: Optional[Tensor] = None
) -> Tensor:
    r"""Isochromats free precession with given relaxation and off-resonance

    This function is only differentiable w.r.t. ``Mi``.

    Setting `T1=T2=None` to opt for simulation ignoring relaxation.

    Usage:
        ``Mo = freeprec(Mi, dur, *, T1, T2, Δf)``
    Inputs:
        - ``Mi``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
          magnitude [0 0 1]
        - ``dur``: `()` ⊻ `(N ⊻ 1,)`, "Sec", duration of free-precession.
    OPTIONALS:
        - ``T1``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T1 relaxation.
        - ``T2``: `()` ⊻ `(N ⊻ 1, *Nd ⊻ 1,)`, "Sec", T2 relaxation.
        - ``Δf``: `(N ⊻ 1, *Nd ⊻ 1,)`, "Hz", off-resonance.
    Outputs:
        - ``Mo``: `(N, *Nd, xyz)`, Result magnetic spins
    """
    ndim = Mi.ndim  # dur, T1, T2, Δf are reshaped to be compatible w/ M
    dur = dur.reshape(dur.shape+(ndim-dur.ndim)*(1,))

    assert((T1 is None) == (T2 is None))  # both or neither
    if T1 is not None:  # → (N ⊻ 1, *Nd ⊻ len(Nd)*(1,), 1)
        T1, T2 = (x.reshape(x.shape+(ndim-x.ndim)*(1,)) for x in (T1, T2))

    if Δf is not None:  # → (N ⊻ 1, *Nd ⊻ len(Nd)*(1,))
        Δf = Δf.reshape(Δf.shape+(ndim-1-Δf.ndim)*(1,))

    return FreePrec.apply(Mi, dur, T1, T2, Δf)
