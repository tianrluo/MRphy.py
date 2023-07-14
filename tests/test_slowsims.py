import numpy as np
import torch
import pytest
from torch import tensor, cuda

from mrphy import γH, dt0, π
from mrphy import beffective, slowsims


class Test_slowsims:

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # dtype, atol = torch.float32, 1e-4
    dtype, atol = torch.float64, 1e-9
    print(device)

    dkw = {'dtype': dtype, 'device': device}

    γ = γH.to(**dkw)  # Hz/Gauss
    dt = dt0.to(**dkw)  # Sec

    @staticmethod
    def np(x):
        return x.detach().cpu().numpy()

    def test_blochsims(self):

        dkw, atol = self.dkw, self.atol
        γ, dt = self.γ, self.dt

        # spins  # (1,nM,xyz)
        M0 = tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], **dkw)
        N, nM, nT = M0.shape[0], M0.shape[1], 512
        Nd = (nM,)

        # parameters: Sec; cm.
        T1, T2 = tensor([[1.]], **dkw), tensor([[4e-2]], **dkw)

        E1, E2, γ2πdt = torch.exp(-dt/T1), torch.exp(-dt/T2), 2*π*γ*dt
        E1_1 = E1 - 1

        loc_x = torch.linspace(-1., 1., steps=nM, **dkw).reshape((N,)+Nd)
        loc_y = torch.linspace(-1., 1., steps=nM, **dkw).reshape((N,)+Nd)
        loc_z = torch.ones((N,)+Nd, **dkw)
        loc = torch.stack([loc_x, loc_y, loc_z], 2)  # (1,nM,xyz)

        Δf = -loc_x * γ  # gr_x==1 Gauss/cm cancels Δf
        b1Map = tensor([1., 0.], **dkw).reshape((N, 1, 2, 1))

        # pulse: Sec; Gauss; Gauss/cm.
        pulse_size = (N, 1, nT)
        t = torch.arange(0, nT, **dkw).reshape(pulse_size)
        rf = 10*torch.cat([torch.cos(t/nT*2*π),              # (1,xy,nT,nCoils)
                           torch.sin(t/nT*2*π)], 1)[..., None]
        gr = torch.cat([torch.ones(pulse_size, **dkw),
                        torch.zeros(pulse_size, **dkw),
                        10*torch.atan(t - round(nT/2))/π], 1)  # (1,xyz,nT)

        rf.requires_grad, gr.requires_grad = True, True

        beff = beffective.rfgr2beff(rf, gr, loc, Δf=Δf, b1Map=b1Map, γ=γ)

        A, B = beffective.beff2ab(beff, E1=E1, E2=E2, γ=γ, dt=dt)

        # sim
        Mo1 = slowsims.blochsim(M0, beff, T1=T1, T2=T2, γ=γ, dt=dt)

        Mo2, Mo_tmp = M0.clone(), M0.clone()
        for t in range(nT):
            Mo2, _ = slowsims.blochsim_1step(Mo2, Mo_tmp, beff[..., t, :],
                                             E1, E1_1, E2, γ2πdt)

        Mo3 = slowsims.blochsim_ab(M0, A, B)

        # assertion
        Mo0 = np.array(
            [[[0.559535641648385,  0.663342640621335, 0.416341441715101],
              [0.391994737048090,  0.210182892388552, -0.860954821972489],
              [-0.677062008711222, 0.673391604920576, -0.143262993311057]]])
        ref = pytest.approx(Mo0, abs=atol)

        f1, f2, f3 = (self.np(x) == ref for x in (Mo1, Mo2, Mo3))
        assert(f1 and f2 and f3)

        # Verify gradients can chain rule back to `rf` and `gr`
        foo = torch.sum(Mo1)
        foo.backward(retain_graph=True)  # keep graph to check `bar.backward()`
        rf_grad1, gr_grad1 = self.np(rf.grad), self.np(gr.grad)
        rf.grad = gr.grad = None  # clear grads from `foo`

        bar = torch.sum(Mo3)
        bar.backward()
        rf_grad2, gr_grad2 = self.np(rf.grad), self.np(gr.grad)
        assert(rf_grad1 == pytest.approx(rf_grad2, abs=atol))
        assert(gr_grad1 == pytest.approx(gr_grad2, abs=atol))

        return

    def test_freeprec(self):

        dkw, atol = self.dkw, self.atol

        # spins  # (1,nM,xyz)
        Mi = tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], **dkw)

        E1, E2 = tensor([[0.5]], **dkw), tensor([[0.5]], **dkw)  # (1,1)

        # scalar dur, T1, T2
        dur = torch.tensor(0.5, **dkw)
        T1, T2 = -dur/torch.log(E1), -dur/torch.log(E2)  # ()

        Δf = tensor([[1/4/dur, -1/4/dur, 1]], **dkw)  # (1, nM) quater-circle

        Mo = slowsims.freeprec(Mi, dur, T1=T1, T2=T2, Δf=Δf)

        Mo0 = np.array([[[0., -0.5, 0.5], [-0.5, 0, 0.5], [0., 0., 1.]]])
        Mref = pytest.approx(Mo0, abs=atol)

        assert(self.np(Mo) == Mref)
        return


if __name__ == '__main__':
    tmp = Test_slowsims()
    tmp.test_blochsims()
    tmp.test_freeprec()
