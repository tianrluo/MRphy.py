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


class BlochSim(Function):
    r"""BlochSim with explict Jacobian operation (backward)
    """

    @staticmethod
    def forward(
            ctx: CTX, Mi: Tensor, Beff: Tensor,
            T1: Optional[Tensor], T2: Optional[Tensor],
            γ: Tensor, dt: Tensor) -> Tensor:
        r"""Forward evolution of Bloch simulation

        Inputs:
            - ``ctx``: `(1,)`, pytorch CTX cacheing object
            - ``Mi``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
              [0 0 1]
            - ``Beff``: `(N, *Nd, xyz, nT)`, "Gauss", B-effective, magnetic \
              field.
        Optionals:
            - ``T1``: `(N, *Nd,)`, "Sec", T1 relaxation.
            - ``T2``: `(N, *Nd,)`, "Sec", T2 relaxation.
            - ``γ``:  `(N, *Nd,)`, "Hz/Gauss", gyro ratio.
            - ``dt``: `(N, 1,)`, "Sec", dwell time.
        Outputs:
            - ``Mo``: `(N, *Nd, xyz)`, Magetic spins after simulation.
        """
        NNd, nT = Beff.shape[:-2], Beff.shape[-1]

        # %% Preprocessing
        γ2πdt = 2*π*γ*dt
        Mi = Mi.clone()  # isolate
        γBeff = γ2πdt*Beff

        assert((T1 is None) == (T2 is None))  # both or neither

        if T1 is None:  # relaxations ignored
            E = e1_1 = None
            fn_relax_ = lambda m1: None  # noqa: E731
        else:
            E1, E2 = torch.exp(-dt/T1), torch.exp(-dt/T2)
            E, e1_1 = torch.cat((E2, E2, E1), dim=-2), E1-1
            fn_relax_ = lambda m1: (m1.mul_(E)  # noqa: E731
                                    )[..., 2:3, :].sub_(e1_1)

        # Pre-allocate intermediate variables, in case of overhead alloc's
        u = γBeff.new_empty(NNd+(3, 1))
        ϕ, cϕ, sϕ = (γBeff.new_empty(NNd+(1, 1)) for _ in range(3))

        # %% Other variables to be cached
        m0, Mhst = Mi[..., None], γBeff.new_empty(NNd+(3, nT))

        # %% Simulation. could we learn to live right.
        for m1, γbeff in zip(Mhst.split(1, dim=-1), γBeff.split(1, dim=-1)):
            # Rotation
            torch.norm(γbeff, dim=-2, keepdim=True, out=ϕ)
            ϕ.clamp_(min=1e-12)
            torch.div(γbeff, ϕ, out=u)

            torch.cos(ϕ, out=cϕ), torch.sin(ϕ, out=sϕ)
            utm0 = ϕ  # ϕ reused uᵀm₀
            torch.sum(u*m0, dim=-2, keepdim=True, out=utm0)
            # wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
            # m₁ = cϕ*m₀ + (1-cϕ)*uᵀm₀*u - sϕ*u×m₀
            torch.cross(u, m0, dim=-2, out=m1)
            m1.mul_(-sϕ).add_(cϕ*m0+(1-cϕ)*utm0*u)  # -sϕ: BxM -> MxB

            # Relaxation
            fn_relax_(m1)

            m0 = m1

        ctx.save_for_backward(Mi, Mhst, γBeff, E, e1_1, γ2πdt)
        Mo = Mhst[..., -1].clone()  # -> (N, *Nd, xyz)
        return Mo

    @staticmethod
    def backward(ctx: CTX, grad_Mo: Tensor
                 ) -> Tuple[Tensor, Tensor, None, None, None, None]:
        r"""Backward evolution of Bloch simulation Jacobians

        Inputs:
            - ``ctx``: `(1,)`, pytorch CTX cacheing object
            - ``grad_Mo``: `(N, *Nd, xyz)`, derivative w.r.t. output Magetic \
              spins.
        Outputs:
            - ``grad_Mi``: `(N, *Nd, xyz)`, derivative w.r.t. input Magetic \
              spins.
            - ``grad_Beff``: `(N,*Nd,xyz,nT)`, derivative w.r.t. B-effective.
            - None*4, this implemendation do not provide derivatives w.r.t.: \
              `T1`, `T2`, `γ`, and `dt`.
        """
        # grads of configuration variables are not supported yet
        needs_grad = ctx.needs_input_grad
        grad_Beff = grad_Mi = grad_T1 = grad_T2 = grad_γ = grad_dt = None

        if not any(needs_grad[0:2]):  # (Mi,Beff;T1,T2,γ,dt):
            return grad_Mi, grad_Beff, grad_T1, grad_T2, grad_γ, grad_dt

        # %% Jacobians. If we turn back time,
        # ctx.save_for_backward(Mhst, γBeff, E, e1_1, γ2πdt)
        Mi, Mhst, γBeff, E, e1_1, γ2πdt = ctx.saved_tensors
        NNd = γBeff.shape[:-2]
        Mi = Mi[..., None]  # (N, *Nd, xyz, 1)

        # assert((E is None) == (e1_1 is None))  # both or neither
        if E is None:  # relaxations ignored
            fn_relax_h1_ = lambda h1: None  # noqa: E731
            fn_relax_m1_ = lambda m1: None  # noqa: E731
        else:
            fn_relax_h1_ = lambda h1: h1.mul_(E)  # noqa: E731

            def fn_relax_m1_(m1):
                m1[..., 2:3, :].add_(e1_1)
                m1.div_(E)
                return

        # Pre-allocate intermediate variables, in case of overhead alloc's
        h0, h1 = γBeff.new_empty(NNd+(3, 1)), (grad_Mo.clone())[..., None]
        uxh1, m0xh1 = (γBeff.new_empty(NNd+(3, 1)) for _ in range(2))
        ϕ, cϕ, sϕ, utm0, uth1 = (γBeff.new_empty(NNd+(1, 1)) for _ in range(5))
        ϕis0 = γBeff.new_empty(NNd+(1, 1), dtype=torch.bool)

        m1 = Mhst.narrow(-1, -1, 1)
        u_dflt = γBeff.new_tensor([[0.], [0.], [1.]])  # (xyz, 1)
        for m0, γbeff in zip(reversed((Mi,)+Mhst.split(1, dim=-1)[:-1]),
                             reversed(γBeff.split(1, dim=-1))):
            # Rotations:
            torch.norm(γbeff, dim=-2, keepdim=True, out=ϕ)
            ϕ.clamp_(min=1e-12)
            γbeff.div_(ϕ)
            u = γbeff  # γbeff reused as u

            torch.logical_not(ϕ, out=ϕis0)
            torch.cos(ϕ, out=cϕ), torch.sin(ϕ, out=sϕ)

            # Resolve singularities
            if torch.any(ϕis0):
                u[ϕis0[..., 0, 0]] = u_dflt  # TODO: Adaptive approach?

            # Relaxation:
            fn_relax_h1_(h1)  # h₁ → h̃₁

            torch.sum(u*h1, dim=-2, keepdim=True, out=uth1)  # uᵀh̃₁
            torch.sum(u*m0, dim=-2, keepdim=True, out=utm0)  # uᵀm₀
            torch.cross(u, h1, dim=-2, out=uxh1)             # u×h̃₁
            torch.cross(m0, h1, dim=-2, out=m0xh1)           # m₀×h̃₁

            # h₀ ≔ ∂L/∂m₀
            # wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
            # h₀ = cϕ*h₁ + (1-cϕ)*uᵀh₁*u + sϕ*u×h₁
            torch.mul(h1, cϕ, out=h0)
            cϕ.sub_(1)  # cϕ → cϕ-1
            h0.add_(sϕ*uxh1-cϕ*uth1*u)

            # ∂L/∂B[..., t]
            cϕ.div_(ϕ), sϕ.div_(ϕ)  # cϕ-1, sϕ → (cϕ-1)/ϕ, sϕ/ϕ
            if torch.any(ϕis0):  # handle division-by-0
                cϕ[ϕis0], sϕ[ϕis0] = 0, 1

            # m₁ → m̃₁ → (m̃₁ - sϕ⋅m₀)⨀(u×h̃₁)
            fn_relax_m1_(m1)
            m1.sub_(sϕ*m0).mul_(uxh1)

            # h̃₁ → (cϕ-1)/ϕ*(uᵀm₀*h̃₁+uᵀh̃₁*m₀)
            h1.mul_(utm0).add_(uth1*m0).mul_(cϕ)

            # (cϕ-1)/ϕ → (2*cϕ*uᵀh̃1*uᵀm₀+(u×h̃₁)ᵀm₁)
            cϕ.mul_(2*uth1*utm0).add_(torch.sum(m1, dim=-2, keepdim=True))

            # u → (-(2*cϕ*uᵀh̃1*uᵀm₀+(u×h̃₁)ᵀm₁)*u
            #      +sϕ/ϕ*m₀×h̃₁
            #      +(cϕ-1)/ϕ*(uᵀm₀*h̃₁+uᵀh̃₁*m₀)) ≡ ∂L/∂B[..., t]
            grad_beff = u  # u (γbeff) is re-reused as grad_beff
            grad_beff.mul_(-cϕ).add_(m0xh1.mul_(sϕ).add_(h1))

            m1, h1, h0 = m0, h0, h1

        # %% Clean up
        grad_Beff = γBeff
        grad_Beff.mul_(-γ2πdt)

        grad_Mi = h1[..., 0] if needs_grad[0] else None
        # forward(ctx, Mi, Beff; T1, T2, γ, dt):
        return grad_Mi, grad_Beff, grad_T1, grad_T2, grad_γ, grad_dt


def blochsim(
        Mi: Tensor, Beff: Tensor,
        T1: Optional[Tensor] = None, T2: Optional[Tensor] = None,
        γ: Tensor = γH, dt: Tensor = dt0) -> Tensor:
    r"""Bloch simulator with explicit Jacobian operation.

    Setting `T1=T2=None` to opt for simulation ignoring relaxation.

    Usage:
        ``Mo = blochsim(Mi, Beff; T1, T2, γ, dt)``
        ``Mo = blochsim(Mi, Beff; T1=None, T2=None, γ, dt)``
    Inputs:
        - ``Mi``: `(N, *Nd, xyz)`, Magnetic spins, assumed equilibrium \
          [[[0 0 1]]].
        - ``Beff``: `(N, *Nd, xyz, nT)`, "Gauss", B-effective, magnetic field.
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

    # (N, *Nd, :, :) compatible for {γ, dt, T1, T2}
    γ, dt = (x.reshape(x.shape+(ndim-x.ndim)*(1,)) for x in (γ, dt))

    assert((T1 is None) == (T2 is None))  # both or neither
    if T1 is not None:
        T1, T2 = (x.reshape(x.shape+(ndim-x.ndim)*(1,)) for x in (T1, T2))

    return BlochSim.apply(Mi, Beff, T1, T2, γ, dt)
